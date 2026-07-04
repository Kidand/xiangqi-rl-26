# -*- coding: utf-8 -*-
"""
GUI FastAPI 后端测试（DESIGN.md §13 §15）。

无需模型验证 hvh 完整流程：
  new / 合法着 / 非法着 400 / undo / export / replay-load（文本 & v1）
  records list/get / 路径穿越拒绝 / model/info

由于 starlette 1.3.1 的 TestClient 依赖 httpx2（环境未安装），
本文件改用：uvicorn 后台线程 + 标准库 urllib.request 的轻量测试客户端。
"""

from __future__ import annotations

import json
import os
import re
import socket
import threading
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any, Optional

import pytest
import uvicorn

# ────────────────────────────────────────────────────────────────
# 轻量测试客户端（urllib 封装，接口类似 requests）
# ────────────────────────────────────────────────────────────────
class _Response:
    def __init__(self, status: int, data: Any):
        self.status_code = status
        self._data = data

    def json(self) -> Any:
        return self._data

    @property
    def ok(self) -> bool:
        return self.status_code < 400


class SimpleClient:
    """极简 HTTP 客户端，包装 urllib，用于测试。"""

    def __init__(self, base_url: str):
        self.base_url = base_url.rstrip("/")

    def _request(
        self,
        method: str,
        path: str,
        body: Optional[dict] = None,
        params: Optional[dict] = None,
    ) -> _Response:
        url = self.base_url + path
        if params:
            from urllib.parse import urlencode
            url += "?" + urlencode(params)

        data = json.dumps(body).encode("utf-8") if body is not None else None
        headers = {"Content-Type": "application/json"} if data else {}

        req = urllib.request.Request(url, data=data, headers=headers, method=method)
        try:
            with urllib.request.urlopen(req) as resp:
                raw = resp.read()
                return _Response(resp.status, json.loads(raw))
        except urllib.error.HTTPError as e:
            raw = e.read()
            try:
                payload = json.loads(raw)
            except Exception:
                payload = {"error": raw.decode(errors="replace")}
            return _Response(e.code, payload)

    def get(self, path: str, params: Optional[dict] = None) -> _Response:
        return self._request("GET", path, params=params)

    def post(self, path: str, body: Optional[dict] = None) -> _Response:
        return self._request("POST", path, body=body)


# ────────────────────────────────────────────────────────────────
# pytest fixture：启动后台服务器
# ────────────────────────────────────────────────────────────────

def _free_port() -> int:
    """找一个空闲端口。"""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


@pytest.fixture(autouse=True)
def records_dir(tmp_path, monkeypatch):
    """把 gui.server._RECORDS_DIR 指向每个用例独立的临时目录。

    硬要求：多个用例会把对局下到终局/认输，从而触发终局自动落盘（§12）；
    没有本 fixture 它们会往仓库真实 records/ 写文件。服务器在请求处理时经模块
    全局 _RECORDS_DIR 取路径，故此 monkeypatch 对后台 uvicorn 线程同样生效。
    records 浏览相关用例（records_get 等）统一改写到本临时目录，方案一致不冲突。
    返回临时 records 根目录，供自动落盘用例定位 records/gui 下的文件。
    """
    import gui.server as srv

    rec_dir = tmp_path / "records"
    rec_dir.mkdir(parents=True, exist_ok=True)
    monkeypatch.setattr(srv, "_RECORDS_DIR", rec_dir)
    return rec_dir


@pytest.fixture(scope="module")
def client():
    """启动测试服务器（无模型），返回 SimpleClient。"""
    from gui.server import app, _games, _model_state

    # 重置全局状态（避免跨测试污染）
    _games.clear()
    _model_state.model = None
    _model_state.path = ""

    port = _free_port()
    config = uvicorn.Config(app, host="127.0.0.1", port=port, log_level="error")
    server = uvicorn.Server(config)
    t = threading.Thread(target=server.run, daemon=True)
    t.start()

    # 等待服务器就绪
    deadline = time.time() + 10.0
    while time.time() < deadline:
        try:
            urllib.request.urlopen(f"http://127.0.0.1:{port}/api/model/info", timeout=1)
            break
        except Exception:
            time.sleep(0.1)

    yield SimpleClient(f"http://127.0.0.1:{port}")

    # 关闭服务器
    server.should_exit = True
    t.join(timeout=3)


# ────────────────────────────────────────────────────────────────
# 辅助函数
# ────────────────────────────────────────────────────────────────
def _new_hvh(client: SimpleClient, fen: Optional[str] = None) -> dict:
    body: dict = {"mode": "hvh", "human_side": "red", "sims": 50}
    if fen:
        body["fen"] = fen
    resp = client.post("/api/game/new", body=body)
    assert resp.status_code == 200, resp.json()
    return resp.json()


def _assert_gamestate_shape(gs: dict) -> None:
    """验证 GameState 必要字段均存在。"""
    required = [
        "game_id", "fen", "side_to_move", "legal_moves",
        "last_move", "in_check", "game_over", "result",
        "termination", "move_history",
    ]
    for key in required:
        assert key in gs, f"GameState 缺少字段: {key!r}"


# ────────────────────────────────────────────────────────────────
# 测试：model/info（无模型）
# ────────────────────────────────────────────────────────────────
def test_model_info_no_model(client: SimpleClient):
    resp = client.get("/api/model/info")
    assert resp.status_code == 200
    info = resp.json()
    assert info["loaded"] is False
    assert "path" in info
    assert "device" in info


# ────────────────────────────────────────────────────────────────
# 测试：hvh 新建对局
# ────────────────────────────────────────────────────────────────
def test_new_hvh_game(client: SimpleClient):
    gs = _new_hvh(client)
    _assert_gamestate_shape(gs)
    assert gs["game_over"] is False
    assert gs["side_to_move"] == "red"
    # 初始局面 44 个合法着
    assert len(gs["legal_moves"]) == 44
    assert gs["move_history"] == []
    assert gs["last_move"] is None


# ────────────────────────────────────────────────────────────────
# 测试：hvh 合法着
# ────────────────────────────────────────────────────────────────
def test_hvh_legal_move(client: SimpleClient):
    gs = _new_hvh(client)
    gid = gs["game_id"]

    # 炮二平五
    resp = client.post(f"/api/game/{gid}/move", body={"move": "h2e2"})
    assert resp.status_code == 200
    gs2 = resp.json()
    _assert_gamestate_shape(gs2)
    assert gs2["last_move"] == "h2e2"
    assert gs2["side_to_move"] == "black"
    assert len(gs2["move_history"]) == 1
    assert gs2["move_history"][0]["move"] == "h2e2"
    # 中文着法非空
    assert gs2["move_history"][0]["chinese"] != ""


# ────────────────────────────────────────────────────────────────
# 测试：非法着返回 400
# ────────────────────────────────────────────────────────────────
def test_hvh_illegal_move_400(client: SimpleClient):
    gs = _new_hvh(client)
    gid = gs["game_id"]

    # a0->a5 非法（车无法一步跳过中间子）
    resp = client.post(f"/api/game/{gid}/move", body={"move": "a0a5"})
    assert resp.status_code == 400
    body = resp.json()
    # 错误信息位于顶层 "error" 字段（DESIGN §13）
    assert "error" in body, f"期望 {{error: ...}} 格式，实际: {body}"


# ────────────────────────────────────────────────────────────────
# 测试：undo（hvh 撤一步）
# ────────────────────────────────────────────────────────────────
def test_hvh_undo(client: SimpleClient):
    gs = _new_hvh(client)
    gid = gs["game_id"]

    # 走一步
    client.post(f"/api/game/{gid}/move", body={"move": "h2e2"})
    client.post(f"/api/game/{gid}/move", body={"move": "h9g7"})

    # 撤销一步（黑方刚走）
    resp = client.post(f"/api/game/{gid}/undo")
    assert resp.status_code == 200
    gs2 = resp.json()
    _assert_gamestate_shape(gs2)
    # 应退回到黑方走之前（回合数减 1）
    assert len(gs2["move_history"]) == 1
    assert gs2["side_to_move"] == "black"


# ────────────────────────────────────────────────────────────────
# 测试：export
# ────────────────────────────────────────────────────────────────
def test_hvh_export(client: SimpleClient):
    gs = _new_hvh(client)
    gid = gs["game_id"]

    client.post(f"/api/game/{gid}/move", body={"move": "h2e2"})

    resp = client.get(f"/api/game/{gid}/export")
    assert resp.status_code == 200
    record = resp.json()
    assert record.get("format") == "xiangqi-record-v1"
    assert "moves" in record
    assert record["moves"] == ["h2e2"]
    assert "fens" in record
    assert len(record["fens"]) >= 2  # 起始 + 至少一步后


# ────────────────────────────────────────────────────────────────
# 测试：replay/load — v1 JSON
# ────────────────────────────────────────────────────────────────
def test_replay_load_v1(client: SimpleClient):
    record = {
        "format": "xiangqi-record-v1",
        "start_fen": "rnbakabnr/9/1c5c1/p1p1p1p1p/9/9/P1P1P1P1P/1C5C1/9/RNBAKABNR w - - 0 1",
        "moves": ["h2e2", "h9g7"],
        "result": None,
    }
    resp = client.post("/api/replay/load", body={"record": record})
    assert resp.status_code == 200
    data = resp.json()
    assert "fens" in data
    assert "moves" in data
    assert "chinese" in data
    assert len(data["fens"]) == 3   # 起始 + 2 步
    assert data["moves"] == ["h2e2", "h9g7"]
    assert len(data["chinese"]) == 2


# ────────────────────────────────────────────────────────────────
# 测试：replay/load — 纯文本格式
# ────────────────────────────────────────────────────────────────
def test_replay_load_text(client: SimpleClient):
    text = "h2e2 h9g7\nb0c2"
    resp = client.post("/api/replay/load", body={"text": text})
    assert resp.status_code == 200
    data = resp.json()
    assert len(data["moves"]) == 3
    assert data["moves"] == ["h2e2", "h9g7", "b0c2"]


# ────────────────────────────────────────────────────────────────
# 测试：records/list
# ────────────────────────────────────────────────────────────────
def test_records_list(client: SimpleClient):
    resp = client.get("/api/records/list")
    assert resp.status_code == 200
    data = resp.json()
    assert "files" in data
    assert isinstance(data["files"], list)
    # 每条记录包含必要字段
    for f in data["files"]:
        assert "path" in f
        assert "games" in f
        assert "mtime" in f


# ────────────────────────────────────────────────────────────────
# 测试：records/get — 基本功能（动态创建测试文件）
# ────────────────────────────────────────────────────────────────
def test_records_get(client: SimpleClient, records_dir: Path):
    """在（已被 monkeypatch 到临时目录的）records/ 下写入一个测试记录文件后通过 API 读取。"""
    import xiangqi.records as rmod
    from xiangqi.constants import START_FEN

    # 使用 autouse fixture 指向的临时 records/ 目录（服务器读的是同一路径）
    rec_dir = records_dir
    rec_dir.mkdir(parents=True, exist_ok=True)

    test_file = rec_dir / "_test_api_get.json"
    record = {
        "format": "xiangqi-record-v1",
        "start_fen": START_FEN,
        "moves": ["h2e2"],
        "result": None,
        "termination": None,
        "plies": 1,
        "red": "test", "black": "test",
        "meta": {},
    }
    rmod.write_record(test_file, record)

    try:
        resp = client.get("/api/records/get", params={"path": "_test_api_get.json", "index": "0"})
        assert resp.status_code == 200
        rec = resp.json()
        assert rec.get("format") == "xiangqi-record-v1"
        assert rec["moves"] == ["h2e2"]
    finally:
        test_file.unlink(missing_ok=True)


# ────────────────────────────────────────────────────────────────
# 测试：路径穿越拒绝
# ────────────────────────────────────────────────────────────────
def test_records_path_traversal_rejected(client: SimpleClient):
    # 尝试访问 records/ 外的文件
    resp = client.get("/api/records/get", params={"path": "../DESIGN.md", "index": "0"})
    # 必须被拒绝（403 或 404）
    assert resp.status_code in (403, 404), (
        f"路径穿越未被拒绝，状态码: {resp.status_code}"
    )


# ────────────────────────────────────────────────────────────────
# 测试：hva/ava 无模型时返回 400
# ────────────────────────────────────────────────────────────────
def test_hva_without_model_returns_400(client: SimpleClient):
    resp = client.post("/api/game/new", body={"mode": "hva", "human_side": "red", "sims": 50})
    assert resp.status_code == 400
    body = resp.json()
    # DESIGN §13: 错误格式 {"error": "..."} 顶层字段，且含中文提示
    assert "error" in body, f"期望 {{error: ...}}，实际: {body}"
    assert body["error"]  # 非空字符串


def test_ava_without_model_returns_400(client: SimpleClient):
    resp = client.post("/api/game/new", body={"mode": "ava", "human_side": "red", "sims": 50})
    assert resp.status_code == 400


# ────────────────────────────────────────────────────────────────
# 测试：game/{id} 不存在时 404
# ────────────────────────────────────────────────────────────────
def test_get_game_not_found(client: SimpleClient):
    resp = client.get("/api/game/nonexistent-id")
    assert resp.status_code == 404


# ────────────────────────────────────────────────────────────────
# 测试：GameState 字段完整性（详细）
# ────────────────────────────────────────────────────────────────
def test_gamestate_all_fields_present(client: SimpleClient):
    """验证 DESIGN §13 规定的 GameState 所有字段均存在。"""
    gs = _new_hvh(client)
    fields = [
        "game_id", "fen", "side_to_move", "legal_moves",
        "last_move", "in_check", "game_over", "result",
        "termination", "move_history", "eval",
    ]
    for f in fields:
        assert f in gs, f"缺少字段: {f!r}，实际: {list(gs.keys())}"


# ────────────────────────────────────────────────────────────────
# 测试：GET /api/game/{id} 路由
# ────────────────────────────────────────────────────────────────
def test_get_game_state(client: SimpleClient):
    gs = _new_hvh(client)
    gid = gs["game_id"]
    resp = client.get(f"/api/game/{gid}")
    assert resp.status_code == 200
    gs2 = resp.json()
    assert gs2["game_id"] == gid
    _assert_gamestate_shape(gs2)


# ────────────────────────────────────────────────────────────────
# 测试：无模型时 with_eval=true 不报错、eval 为 null
# （置于无模型区段：client_with_model 会把模型加载进共享的 _model_state 全局，
#   一旦加载后 client 服务器也可见，故此用例必须在任何模型用例之前运行。）
# ────────────────────────────────────────────────────────────────
def test_with_eval_no_model_returns_null(client: SimpleClient):
    """无模型时 with_eval=true 不报错，eval 为 null。"""
    gs0 = _new_hvh(client)
    gid = gs0["game_id"]
    resp = client.get(f"/api/game/{gid}", params={"with_eval": "true"})
    assert resp.status_code == 200
    gs = resp.json()
    _assert_gamestate_shape(gs)
    assert gs.get("eval") is None, f"无模型时 eval 应为 null，实际 {gs.get('eval')}"


# ────────────────────────────────────────────────────────────────
# 测试：hvh 走子后 undo 再 undo（边界：只有 1 步时不报错）
# ────────────────────────────────────────────────────────────────
def test_undo_boundary(client: SimpleClient):
    gs = _new_hvh(client)
    gid = gs["game_id"]

    # 未走任何步时 undo 应返回 400
    resp = client.post(f"/api/game/{gid}/undo")
    assert resp.status_code == 400

    # 走一步后可以 undo
    client.post(f"/api/game/{gid}/move", body={"move": "h2e2"})
    resp2 = client.post(f"/api/game/{gid}/undo")
    assert resp2.status_code == 200
    gs2 = resp2.json()
    assert len(gs2["move_history"]) == 0


# ────────────────────────────────────────────────────────────────
# 测试：replay/load 非法着法返回 400
# ────────────────────────────────────────────────────────────────
def test_replay_load_illegal_move(client: SimpleClient):
    from xiangqi.constants import START_FEN
    record = {
        "format": "xiangqi-record-v1",
        "start_fen": START_FEN,
        "moves": ["a0a5"],  # 非法着法
    }
    resp = client.post("/api/replay/load", body={"record": record})
    assert resp.status_code == 400


# ────────────────────────────────────────────────────────────────
# 测试：自定义 FEN 建局
# ────────────────────────────────────────────────────────────────
def test_new_game_custom_fen(client: SimpleClient):
    # 简单残局局面
    fen = "4k4/9/9/9/9/9/9/9/9/4K4 w - - 0 1"
    gs = _new_hvh(client, fen=fen)
    assert gs["fen"] == fen
    assert gs["game_over"] is False


# ────────────────────────────────────────────────────────────────
# 测试：非法 FEN 建局返回 400
# ────────────────────────────────────────────────────────────────
def test_new_game_bad_fen(client: SimpleClient):
    resp = client.post("/api/game/new", body={"mode": "hvh", "fen": "invalid_fen", "sims": 50})
    assert resp.status_code == 400


# ────────────────────────────────────────────────────────────────
# 模型相关：fixture（需 torch；用极小模型加快速度）
# ────────────────────────────────────────────────────────────────

@pytest.fixture(scope="module")
def client_with_model(tmp_path_factory):
    """启动带极小模型的测试服务器，用于验证 AI / eval 相关端点。"""
    import torch
    from rl.model import create_model, save_checkpoint
    from rl.config import ModelConfig
    from gui.server import app, _games, _model_state

    tmp = tmp_path_factory.mktemp("models")
    ckpt_path = str(tmp / "mini.pt")
    cfg = ModelConfig(blocks=1, filters=16)
    m = create_model(cfg)
    save_checkpoint(m, cfg, ckpt_path, iteration=0)

    # 重置全局状态
    _games.clear()
    _model_state.model = None
    _model_state.path = ""

    port = _free_port()
    config = uvicorn.Config(app, host="127.0.0.1", port=port, log_level="error")
    server = uvicorn.Server(config)
    t = threading.Thread(target=server.run, daemon=True)
    t.start()

    deadline = time.time() + 10.0
    while time.time() < deadline:
        try:
            urllib.request.urlopen(f"http://127.0.0.1:{port}/api/model/info", timeout=1)
            break
        except Exception:
            time.sleep(0.1)

    c = SimpleClient(f"http://127.0.0.1:{port}")
    # 加载模型
    resp = c.post("/api/model/load", body={"path": ckpt_path})
    assert resp.status_code == 200, f"模型加载失败: {resp.json()}"

    yield c

    server.should_exit = True
    t.join(timeout=3)


# ────────────────────────────────────────────────────────────────
# 测试：hva eval 语义（评估条符号契约，DESIGN §13）
# 新契约：eval 对象带显式 "side"（"red"|"black"），声明 value 是哪一方视角
#   （= MCTS 运行时的走子方）。AI 落子附带的 eval：side = 落子前的走子方（AI 方）。
#   前端据 eval.side 换算红方视角（side=="red" 取 value，否则取 -value），
#   不再靠 side_to_move 反推（历史上出过符号颠倒 bug 的推断链）。
# ────────────────────────────────────────────────────────────────

def test_hva_eval_side_is_ai_human_red(client_with_model: SimpleClient):
    """hva human=red：AI 执黑，回着后 eval.side 恒为 "black"（AI 方），value ∈ [-1,1]。"""
    resp = client_with_model.post("/api/game/new",
                                  body={"mode": "hva", "human_side": "red", "sims": 4})
    assert resp.status_code == 200
    gs0 = resp.json()
    gid = gs0["game_id"]
    legal = gs0.get("legal_moves", [])
    assert legal, "初始局面无合法着法"

    resp2 = client_with_model.post(f"/api/game/{gid}/move", body={"move": legal[0]})
    assert resp2.status_code == 200
    gs = resp2.json()

    # 人类（红）走后 AI（黑）应已回着 → move_history 应有 ≥2 条
    hist = gs.get("move_history", [])
    if not gs.get("game_over"):
        assert len(hist) >= 2, f"AI 未回着，move_history={hist}"
        # 轮回红方
        assert gs["side_to_move"] == "red", f"预期轮到红方，实际 {gs['side_to_move']}"

    # eval 应存在且 value ∈ [-1,1]
    ev = gs.get("eval")
    assert ev is not None, "AI 回着后 GameState 缺少 eval 字段"
    v = ev.get("value")
    assert v is not None, "eval 缺少 value 字段"
    assert -1.0 <= float(v) <= 1.0, f"eval.value={v} 超出 [-1,1]"

    # 核心断言：eval.side 恒等于 AI 方（黑），与 side_to_move（已切回红）无关。
    assert ev.get("side") == "black", f"预期 eval.side=black（AI 方），实际 {ev.get('side')}"
    assert ev.get("top_moves") is not None, "eval 缺少 top_moves 字段"


def test_hva_eval_side_is_ai_human_black(client_with_model: SimpleClient):
    """hva human=black：AI 执红先走，eval.side 恒为 "red"（AI 方），value ∈ [-1,1]。"""
    resp = client_with_model.post("/api/game/new",
                                  body={"mode": "hva", "human_side": "black", "sims": 4})
    assert resp.status_code == 200
    gs = resp.json()

    # AI=红先走 → move_history 应有 ≥1 条，side_to_move 应为 "black"
    hist = gs.get("move_history", [])
    if not gs.get("game_over"):
        assert len(hist) >= 1, f"AI 未先走，move_history={hist}"
        assert gs["side_to_move"] == "black", f"预期轮到黑方，实际 {gs['side_to_move']}"

    ev = gs.get("eval")
    assert ev is not None, "AI 先走后 GameState 缺少 eval 字段"
    v = ev.get("value")
    assert v is not None, "eval 缺少 value 字段"
    assert -1.0 <= float(v) <= 1.0, f"eval.value={v} 超出 [-1,1]"

    # 核心断言：eval.side 恒等于 AI 方（红），与 side_to_move（黑）无关。
    assert ev.get("side") == "red", f"预期 eval.side=red（AI 方），实际 {ev.get('side')}"

    top = ev.get("top_moves", [])
    # top_moves 应按概率降序排列
    for i in range(len(top) - 1):
        assert top[i]["prob"] >= top[i+1]["prob"] - 1e-9, (
            f"top_moves 未降序：[{i}].prob={top[i]['prob']} > [{i+1}].prob={top[i+1]['prob']}"
        )


def test_with_eval_side_is_side_to_move(client_with_model: SimpleClient):
    """GET /api/game/{id}?with_eval=true：eval.side == 当前 side_to_move，value ∈ [-1,1]。"""
    resp = client_with_model.post("/api/game/new",
                                  body={"mode": "hvh", "human_side": "red", "sims": 4})
    assert resp.status_code == 200
    gs0 = resp.json()
    gid = gs0["game_id"]

    resp2 = client_with_model.get(f"/api/game/{gid}",
                                  params={"with_eval": "true", "sims": "4"})
    assert resp2.status_code == 200
    gs = resp2.json()
    ev = gs.get("eval")
    assert ev is not None, "with_eval=true 有模型未终局时 eval 不应为 null"
    v = ev.get("value")
    assert v is not None and -1.0 <= float(v) <= 1.0, f"eval.value={v} 超出 [-1,1]"
    # 未走子 → side_to_move 仍为 red；eval.side 应等于当前 side_to_move
    assert ev.get("side") == gs["side_to_move"], (
        f"eval.side={ev.get('side')} 应等于 side_to_move={gs['side_to_move']}"
    )
    assert gs["side_to_move"] == "red"

    # 走一步后轮到黑方，eval.side 必须跟着变为 black——
    # 防止实现硬编码 side="red" 仍能通过初始局面的断言
    resp3 = client_with_model.post(f"/api/game/{gid}/move", body={"move": "h2e2"})
    assert resp3.status_code == 200
    resp4 = client_with_model.get(f"/api/game/{gid}",
                                  params={"with_eval": "true", "sims": "4"})
    assert resp4.status_code == 200
    gs2 = resp4.json()
    ev2 = gs2.get("eval")
    assert ev2 is not None
    assert gs2["side_to_move"] == "black"
    assert ev2.get("side") == "black", (
        f"黑方待走时 eval.side={ev2.get('side')} 应为 black"
    )


def test_hint_value_in_range(client_with_model: SimpleClient):
    """hint 返回的 value ∈ [-1,1] 且 top_moves 概率降序。"""
    resp = client_with_model.post("/api/game/new",
                                  body={"mode": "hvh", "human_side": "red", "sims": 4})
    assert resp.status_code == 200
    gid = resp.json()["game_id"]

    resp2 = client_with_model.get(f"/api/game/{gid}/hint", params={"sims": "4"})
    assert resp2.status_code == 200
    h = resp2.json()
    assert "move" in h and "value" in h and "top_moves" in h
    assert -1.0 <= float(h["value"]) <= 1.0, f"hint value={h['value']} out of range"
    # DESIGN §13：hint 的 value 必带 side 声明视角（初始局面红方待走）
    assert h.get("side") == "red", f"hint.side={h.get('side')} 应为当前走子方 red"
    tops = h["top_moves"]
    for i in range(len(tops) - 1):
        assert tops[i]["prob"] >= tops[i+1]["prob"] - 1e-9, (
            f"hint top_moves 未降序 [{i}]={tops[i]['prob']} [{i+1}]={tops[i+1]['prob']}"
        )


# ────────────────────────────────────────────────────────────────
# 测试：resign 端点
# ────────────────────────────────────────────────────────────────
def test_resign(client: SimpleClient):
    gs = _new_hvh(client)
    gid = gs["game_id"]

    resp = client.post(f"/api/game/{gid}/resign")
    assert resp.status_code == 200
    gs2 = resp.json()
    assert gs2["game_over"] is True
    assert gs2["result"] in ("1-0", "0-1")
    assert gs2["termination"] == "resign"


# ────────────────────────────────────────────────────────────────
# 测试：终局自动落盘（DESIGN §12）
# records_dir fixture（autouse）已把 gui.server._RECORDS_DIR 指向临时目录。
# ────────────────────────────────────────────────────────────────
_RESULT_ZH = {"1-0": "红胜", "0-1": "黑胜", "1/2-1/2": "和棋"}


def test_autosave_on_resign_creates_one_file(client: SimpleClient, records_dir: Path):
    """(a) hvh 认输 → records/gui 下恰好一个文件；文件名格式正确、胜负与认输方相反、与 result 一致。"""
    gs = _new_hvh(client)   # human_side=red，初始红方待走
    gid = gs["game_id"]

    resp = client.post(f"/api/game/{gid}/resign")
    assert resp.status_code == 200
    result = resp.json()["result"]
    # 红方（初始走子方）认输 → 黑胜
    assert result == "0-1", f"预期红方认输→黑胜(0-1)，实际 {result}"

    gui_dir = records_dir / "gui"
    files = sorted(gui_dir.glob("*.json"))
    assert len(files) == 1, f"认输后应恰好一个文件，实际 {files}"

    name = files[0].name
    m = re.match(r"^\d{8}-\d{6}_hvh_(红胜|黑胜)\.json$", name)
    assert m, f"文件名不匹配约定格式: {name!r}"
    zh = m.group(1)
    # 胜负与认输方相反（红认输→黑胜）且与 GameState.result 一致
    assert zh == "黑胜", f"红方认输文件名应为黑胜，实际 {zh}"
    assert zh == _RESULT_ZH[result], f"文件名胜负 {zh} 与 result {result} 不一致"


def test_autosave_content_readable_and_matches_history(client: SimpleClient, records_dir: Path):
    """(b) 落盘文件能被 read_records 读回，moves 与 move_history 一致，format 正确。"""
    from xiangqi.records import read_records

    gs = _new_hvh(client)
    gid = gs["game_id"]
    client.post(f"/api/game/{gid}/move", body={"move": "h2e2"})
    client.post(f"/api/game/{gid}/move", body={"move": "h9g7"})
    resp = client.post(f"/api/game/{gid}/resign")
    assert resp.status_code == 200
    gs_final = resp.json()

    gui_dir = records_dir / "gui"
    files = sorted(gui_dir.glob("*.json"))
    assert len(files) == 1, f"应恰好一个文件，实际 {files}"

    recs = read_records(files[0])
    assert len(recs) == 1
    rec = recs[0]
    assert rec["format"] == "xiangqi-record-v1"
    expected_moves = [e["move"] for e in gs_final["move_history"]]
    assert rec["moves"] == expected_moves, f"落盘 moves {rec['moves']} != 历史 {expected_moves}"
    assert rec["moves"] == ["h2e2", "h9g7"]


def test_autosave_undo_then_resign_replaces_file(client: SimpleClient, records_dir: Path):
    """(c) 悔棋（撤销认输）后再认输 → 仍只有一个文件（旧文件被替换）。"""
    gs = _new_hvh(client)
    gid = gs["game_id"]

    resp1 = client.post(f"/api/game/{gid}/resign")
    assert resp1.status_code == 200
    gui_dir = records_dir / "gui"
    assert len(list(gui_dir.glob("*.json"))) == 1

    # 悔棋：撤销认输，对局恢复未终局
    resp2 = client.post(f"/api/game/{gid}/undo")
    assert resp2.status_code == 200
    assert resp2.json()["game_over"] is False

    # 再次认输
    resp3 = client.post(f"/api/game/{gid}/resign")
    assert resp3.status_code == 200

    files = list(gui_dir.glob("*.json"))
    assert len(files) == 1, f"再次认输后应仍只有一个文件（旧文件被替换），实际 {files}"


def test_autosave_not_written_when_game_not_over(client: SimpleClient, records_dir: Path):
    """(d) 未终局的对局不产生任何落盘文件。"""
    gs = _new_hvh(client)
    gid = gs["game_id"]
    resp = client.post(f"/api/game/{gid}/move", body={"move": "h2e2"})
    assert resp.status_code == 200
    assert resp.json()["game_over"] is False

    gui_dir = records_dir / "gui"
    files = list(gui_dir.glob("*.json")) if gui_dir.exists() else []
    assert files == [], f"未终局不应产生文件，实际 {files}"


# ────────────────────────────────────────────────────────────────
# 测试：后台实时日志（AI 胜率打印到 stdout）
# ────────────────────────────────────────────────────────────────
def test_value_to_red_pct():
    """胜率换算函数：走子方视角 value → 红方胜率百分比。"""
    from gui.server import _value_to_red_pct
    from xiangqi.constants import RED, BLACK

    # value=+0.264 红走 → 红方胜率 63.2%
    assert abs(_value_to_red_pct(0.264, RED) - 63.2) < 1e-6
    # value=+0.264 黑走（走子方黑方视角 +0.264 → 红方视角 -0.264）→ 红方胜率 36.8%
    assert abs(_value_to_red_pct(0.264, BLACK) - 36.8) < 1e-6

    # 边界：value=0 → 50%；value=+1 → 100%；value=-1 → 0%
    assert abs(_value_to_red_pct(0.0, RED) - 50.0) < 1e-9
    assert abs(_value_to_red_pct(1.0, RED) - 100.0) < 1e-9
    assert abs(_value_to_red_pct(-1.0, RED) - 0.0) < 1e-9


def test_hva_ai_move_logs_eval_to_stdout(client_with_model: SimpleClient, capsys):
    """hva 模式 AI 自动回着后，终端应打印一行含「红方胜率」与百分号的日志。"""
    resp = client_with_model.post(
        "/api/game/new", body={"mode": "hva", "human_side": "red", "sims": 4}
    )
    assert resp.status_code == 200
    gs0 = resp.json()
    gid = gs0["game_id"]
    legal = gs0.get("legal_moves", [])
    assert legal, "初始局面无合法着法"

    resp2 = client_with_model.post(f"/api/game/{gid}/move", body={"move": legal[0]})
    assert resp2.status_code == 200

    captured = capsys.readouterr()
    assert "AI落子" in captured.out, f"日志缺少 AI落子 标签，实际输出: {captured.out!r}"
    assert "红方胜率" in captured.out, f"日志缺少 红方胜率，实际输出: {captured.out!r}"
    assert "%" in captured.out, f"日志缺少百分号，实际输出: {captured.out!r}"


def test_ava_ai_move_logs_eval_to_stdout(client_with_model: SimpleClient, capsys):
    """ava 模式 /ai_move 单步推进后，终端应打印 AI 判断日志。"""
    resp = client_with_model.post(
        "/api/game/new", body={"mode": "ava", "human_side": "red", "sims": 4}
    )
    assert resp.status_code == 200
    gid = resp.json()["game_id"]

    resp2 = client_with_model.post(f"/api/game/{gid}/ai_move", body={"sims": 4})
    assert resp2.status_code == 200

    captured = capsys.readouterr()
    assert "AI落子" in captured.out
    assert "红方胜率" in captured.out
    assert "%" in captured.out


def test_ava_export_both_sides_ai(client_with_model: SimpleClient):
    """ava 模式导出：双方均应标为 "ai"（历史 bug：红方残留 "human"）。"""
    resp = client_with_model.post(
        "/api/game/new", body={"mode": "ava", "human_side": "red", "sims": 4}
    )
    assert resp.status_code == 200
    gid = resp.json()["game_id"]

    # 推进一步（不是必须，但更贴近真实导出场景）
    client_with_model.post(f"/api/game/{gid}/ai_move", body={"sims": 4})

    resp2 = client_with_model.get(f"/api/game/{gid}/export")
    assert resp2.status_code == 200
    record = resp2.json()
    assert record["red"] == "ai", f"ava 红方应为 ai，实际 {record['red']}"
    assert record["black"] == "ai", f"ava 黑方应为 ai，实际 {record['black']}"
    assert record["meta"]["mode"] == "ava"


def test_hint_logs_eval_to_stdout(client_with_model: SimpleClient, capsys):
    """hint 端点应打印含「提示」标签的判断日志（无落子，只报胜率与推荐着）。"""
    resp = client_with_model.post(
        "/api/game/new", body={"mode": "hvh", "human_side": "red", "sims": 4}
    )
    assert resp.status_code == 200
    gid = resp.json()["game_id"]

    resp2 = client_with_model.get(f"/api/game/{gid}/hint", params={"sims": "4"})
    assert resp2.status_code == 200

    captured = capsys.readouterr()
    assert "提示" in captured.out, f"日志缺少 提示 标签，实际输出: {captured.out!r}"
    assert "红方胜率" in captured.out
    assert "%" in captured.out


def test_resign_logs_game_over_to_stdout(client_with_model: SimpleClient, capsys):
    """认输应触发对局结束日志。"""
    resp = client_with_model.post(
        "/api/game/new", body={"mode": "hvh", "human_side": "red", "sims": 4}
    )
    gid = resp.json()["game_id"]

    resp2 = client_with_model.post(f"/api/game/{gid}/resign")
    assert resp2.status_code == 200

    captured = capsys.readouterr()
    assert "对局结束" in captured.out
    assert "termination=resign" in captured.out


def test_log_eval_never_raises_on_bad_input():
    """_log_eval 内部异常必须被吞掉（try/except 包裹），不得向外抛出以打断接口响应。"""
    from gui.server import _log_eval, _GameData
    from xiangqi.board import Board

    game = _GameData(game_id="deadbeef", board=Board(), mode="hvh", human_side="red", sims=10)

    # 故意传入会在内部触发异常的参数（value_mover=None → 算术 TypeError）
    _log_eval(game, "AI落子", "h2e2", "炮二平五", None, 1, 10, 100.0, [{"move": "x", "chinese": "y", "prob": 0.1}])  # 不应抛出
    # top_moves 内含非法结构（非 dict）→ AttributeError（.get 不存在）
    _log_eval(game, "提示", "h2e2", "炮二平五", 0.1, 1, 10, 5.0, ["not-a-dict"])  # 不应抛出


def test_log_game_over_never_raises_when_not_over():
    """_log_game_over 在对局未结束时应静默返回（不打印、不抛出）。"""
    from gui.server import _log_game_over, _GameData
    from xiangqi.board import Board

    game = _GameData(game_id="cafebabe", board=Board(), mode="hvh", human_side="red", sims=10)
    _log_game_over(game)  # 未结束 → 不应抛出


# ────────────────────────────────────────────────────────────────
# 测试：每步胜率 evals（DESIGN §12/§13）
# ────────────────────────────────────────────────────────────────
def test_record_position_eval_perspective_and_sims_priority():
    """走子方视角→红方视角换算；低 sims 不覆盖高 sims。"""
    from gui.server import _record_position_eval, _GameData
    from xiangqi.board import Board
    from xiangqi.constants import RED, BLACK

    game = _GameData(game_id="x", board=Board(), mode="hvh", human_side="red", sims=10)
    _record_position_eval(game, 0.5, BLACK, 100)
    assert game.evals[0] == {"value_red": -0.5, "sims": 100, "source": "play"}
    # 更低 sims 不覆盖
    _record_position_eval(game, 0.9, BLACK, 50)
    assert game.evals[0]["value_red"] == -0.5
    # 更高 sims 覆盖；红方视角原样保存
    _record_position_eval(game, 0.9, RED, 200)
    assert game.evals[0] == {"value_red": 0.9, "sims": 200, "source": "play"}


def test_autosave_evals_length_and_result_slot(client: SimpleClient, records_dir: Path):
    """hvh 无模型：落盘 evals 与局面数等长、人类步为 null、终局槽按 result 映射。"""
    gs = _new_hvh(client)
    gid = gs["game_id"]
    client.post(f"/api/game/{gid}/move", body={"move": "h2e2"})
    # hvh 当前走子方（黑）认输 → 红胜 1-0
    resp = client.post(f"/api/game/{gid}/resign")
    assert resp.status_code == 200

    files = list((records_dir / "gui").glob("*.json"))
    assert len(files) == 1
    rec = json.loads(files[0].read_text(encoding="utf-8"))
    assert rec["result"] == "1-0"
    assert len(rec["evals"]) == rec["plies"] + 1 == 2
    assert rec["evals"][0] is None
    assert rec["evals"][-1] == {"value_red": 1.0, "sims": 0, "source": "result"}


def test_hva_export_evals_ai_positions_filled(client_with_model: SimpleClient):
    """hva：AI 回着评估以 source=play 写入落子前局面槽位，人走局面为 null。"""
    resp = client_with_model.post(
        "/api/game/new", body={"mode": "hva", "human_side": "red", "sims": 4}
    )
    gid = resp.json()["game_id"]
    resp2 = client_with_model.post(f"/api/game/{gid}/move", body={"move": "h2e2"})
    assert resp2.status_code == 200
    assert resp2.json().get("ai_move")  # AI 已回着

    rec = client_with_model.get(f"/api/game/{gid}/export").json()
    evals = rec["evals"]
    assert len(evals) == len(rec["moves"]) + 1 == 3
    assert evals[0] is None            # 人走前局面无评估
    e1 = evals[1]                      # 人走后、AI（黑）落子前的局面
    assert e1 is not None
    assert e1["source"] == "play" and e1["sims"] == 4
    assert -1.0 <= e1["value_red"] <= 1.0
    assert evals[2] is None            # AI 落子后局面尚未评估且未终局


def test_undo_truncates_evals(client_with_model: SimpleClient):
    """hva 悔棋撤两步后 evals 同步截断，与局面序列等长。"""
    resp = client_with_model.post(
        "/api/game/new", body={"mode": "hva", "human_side": "red", "sims": 4}
    )
    gid = resp.json()["game_id"]
    client_with_model.post(f"/api/game/{gid}/move", body={"move": "h2e2"})
    resp2 = client_with_model.post(f"/api/game/{gid}/undo")
    assert resp2.status_code == 200
    assert resp2.json()["move_history"] == []

    rec = client_with_model.get(f"/api/game/{gid}/export").json()
    assert rec["moves"] == []
    assert rec["evals"] == [None]


def test_replay_load_returns_normalized_evals(client: SimpleClient):
    """replay/load 透传规范化 evals；无 evals 记录 → 全 null。"""
    from xiangqi.constants import START_FEN as SF

    good = {"value_red": 0.3, "sims": 64, "source": "play"}
    record = {"format": "xiangqi-record-v1", "start_fen": SF,
              "moves": ["h2e2", "h9g7"], "evals": [good, "bad", None, "overflow"]}
    resp = client.post("/api/replay/load", body={"record": record})
    assert resp.status_code == 200
    data = resp.json()
    assert data["evals"] == [good, None, None]

    resp2 = client.post("/api/replay/load", body={"record": {
        "format": "xiangqi-record-v1", "start_fen": SF, "moves": ["h2e2"]}})
    assert resp2.json()["evals"] == [None, None]


def test_records_eval_no_model_400(client: SimpleClient, records_dir: Path, monkeypatch):
    """无模型时补算端点返回 400。"""
    import gui.server as srv
    from xiangqi.constants import START_FEN as SF

    # client_with_model fixture 若已创建，模块全局已有模型 —— 临时清空
    monkeypatch.setattr(srv._model_state, "model", None)
    gui_dir = records_dir / "gui"
    gui_dir.mkdir(parents=True, exist_ok=True)
    (gui_dir / "t.json").write_text(json.dumps(
        {"format": "xiangqi-record-v1", "start_fen": SF, "moves": ["h2e2"]}
    ), encoding="utf-8")
    resp = client.post("/api/records/eval", body={"path": "gui/t.json"})
    assert resp.status_code == 400
    assert "error" in resp.json()


def test_records_eval_path_traversal_403(client_with_model: SimpleClient):
    """补算端点沿用 records/ 路径穿越防护。"""
    resp = client_with_model.post(
        "/api/records/eval", body={"path": "../../pyproject.toml"}
    )
    assert resp.status_code == 403


def test_records_eval_batched_compute_and_writeback(
    client_with_model: SimpleClient, records_dir: Path
):
    """分批补算：终局槽按 result 免额度映射；每批限 max_positions；随批写回文件。"""
    from xiangqi.constants import START_FEN as SF

    gui_dir = records_dir / "gui"
    gui_dir.mkdir(parents=True, exist_ok=True)
    rec = {"format": "xiangqi-record-v1", "start_fen": SF,
           "moves": ["h2e2", "h9g7"], "result": "1-0", "plies": 2}
    (gui_dir / "t.json").write_text(json.dumps(rec), encoding="utf-8")

    # 第一批：只补 1 个局面；末局面按 result 免额度映射
    resp = client_with_model.post("/api/records/eval", body={
        "path": "gui/t.json", "index": 0, "sims": 4, "max_positions": 1})
    assert resp.status_code == 200, resp.json()
    data = resp.json()
    assert data["total"] == 3
    assert data["pending"] == 1
    e0 = data["evals"][0]
    assert e0["source"] == "replay" and e0["sims"] == 4
    assert -1.0 <= e0["value_red"] <= 1.0
    assert data["evals"][1] is None
    assert data["evals"][2] == {"value_red": 1.0, "sims": 0, "source": "result"}

    # 已写回磁盘
    on_disk = json.loads((gui_dir / "t.json").read_text(encoding="utf-8"))
    assert on_disk["evals"][0]["source"] == "replay"
    assert on_disk["evals"][1] is None

    # 第二批补完剩余局面；再次调用应为幂等空批
    resp2 = client_with_model.post("/api/records/eval", body={
        "path": "gui/t.json", "index": 0, "sims": 4, "max_positions": 8})
    data2 = resp2.json()
    assert data2["pending"] == 0
    assert all(e is not None for e in data2["evals"])
    resp3 = client_with_model.post("/api/records/eval", body={
        "path": "gui/t.json", "index": 0, "sims": 4, "max_positions": 8})
    assert resp3.json() == data2


# ────────────────────────────────────────────────────────────────
# 测试：两段式走子 ai_reply（DESIGN §13）
# ────────────────────────────────────────────────────────────────
def test_move_ai_reply_false_two_phase(client_with_model: SimpleClient):
    """ai_reply=false 只落人类子立即返回；随后 /ai_move 取 AI 回着。"""
    resp = client_with_model.post(
        "/api/game/new", body={"mode": "hva", "human_side": "red", "sims": 4}
    )
    gid = resp.json()["game_id"]

    r1 = client_with_model.post(
        f"/api/game/{gid}/move", body={"move": "h2e2", "ai_reply": False}
    )
    assert r1.status_code == 200
    gs1 = r1.json()
    assert gs1.get("ai_move") is None
    assert len(gs1["move_history"]) == 1
    assert gs1["side_to_move"] == "black"

    r2 = client_with_model.post(f"/api/game/{gid}/ai_move", body={"sims": 4})
    assert r2.status_code == 200
    gs2 = r2.json()
    assert len(gs2["move_history"]) == 2
    assert gs2["side_to_move"] == "red"
    assert gs2["eval"]["side"] == "black"   # AI（黑）落子前的走子方视角


def test_move_ai_reply_default_unchanged(client_with_model: SimpleClient):
    """不带 ai_reply（旧客户端）：hva 行为不变，响应含 AI 回着。"""
    resp = client_with_model.post(
        "/api/game/new", body={"mode": "hva", "human_side": "red", "sims": 4}
    )
    gid = resp.json()["game_id"]
    r = client_with_model.post(f"/api/game/{gid}/move", body={"move": "h2e2"})
    gs = r.json()
    assert gs.get("ai_move")
    assert len(gs["move_history"]) == 2


def test_new_game_ai_reply_false_no_first_move(client_with_model: SimpleClient):
    """AI 执红 + ai_reply=false：新局不含首着；随后 /ai_move 补上。"""
    resp = client_with_model.post("/api/game/new", body={
        "mode": "hva", "human_side": "black", "sims": 4, "ai_reply": False})
    gs = resp.json()
    assert gs["move_history"] == []
    assert gs["side_to_move"] == "red"
    r2 = client_with_model.post(f"/api/game/{gs['game_id']}/ai_move", body={"sims": 4})
    assert r2.status_code == 200
    assert len(r2.json()["move_history"]) == 1


def test_ai_move_rejected_on_human_turn_hva(client_with_model: SimpleClient):
    """hva 轮到人类时 /ai_move 返回 409（两段式 undo 竞态守卫）。"""
    resp = client_with_model.post(
        "/api/game/new", body={"mode": "hva", "human_side": "red", "sims": 4}
    )
    gid = resp.json()["game_id"]
    r = client_with_model.post(f"/api/game/{gid}/ai_move", body={"sims": 4})
    assert r.status_code == 409
    assert "error" in r.json()


# ────────────────────────────────────────────────────────────────
# 测试：两段式窗口的悔棋与轮次守卫（审查修复）
# ────────────────────────────────────────────────────────────────
def test_undo_one_step_in_two_phase_window(client_with_model: SimpleClient):
    """两段式窗口（人类刚落子、AI 未回着）悔棋只撤一步，撤后轮到人类。"""
    resp = client_with_model.post(
        "/api/game/new", body={"mode": "hva", "human_side": "red", "sims": 4}
    )
    gid = resp.json()["game_id"]
    client_with_model.post(f"/api/game/{gid}/move", body={"move": "h2e2", "ai_reply": False})

    r = client_with_model.post(f"/api/game/{gid}/undo")
    assert r.status_code == 200
    gs = r.json()
    assert len(gs["move_history"]) == 0
    assert gs["side_to_move"] == "red"   # 恒撤回到人类轮次


def test_undo_two_steps_after_ai_replied(client_with_model: SimpleClient):
    """AI 已回着的正常态：悔棋撤一整回合两步（旧契约不变）。"""
    resp = client_with_model.post(
        "/api/game/new", body={"mode": "hva", "human_side": "red", "sims": 4}
    )
    gid = resp.json()["game_id"]
    client_with_model.post(f"/api/game/{gid}/move", body={"move": "h2e2"})  # 默认含 AI 回着

    r = client_with_model.post(f"/api/game/{gid}/undo")
    gs = r.json()
    assert len(gs["move_history"]) == 0
    assert gs["side_to_move"] == "red"


def test_human_move_rejected_on_ai_turn(client_with_model: SimpleClient):
    """两段式窗口内（轮到 AI）人类再走子返回 409。"""
    resp = client_with_model.post(
        "/api/game/new", body={"mode": "hva", "human_side": "red", "sims": 4}
    )
    gid = resp.json()["game_id"]
    client_with_model.post(f"/api/game/{gid}/move", body={"move": "h2e2", "ai_reply": False})

    # 此刻轮到黑（AI）——人类不得再走（否则会替 AI 落子）
    r = client_with_model.post(f"/api/game/{gid}/move", body={"move": "h9g7"})
    assert r.status_code == 409
    assert "error" in r.json()


def test_normalize_evals_rejects_nan_and_out_of_range():
    """NaN 与越界 value_red 置 None（防前端 JSON 解析失败/225% 胜率）。"""
    from xiangqi.records import normalize_evals
    rec = {"moves": ["h2e2", "h9g7"], "evals": [
        {"value_red": float("nan"), "sims": 4, "source": "replay"},
        {"value_red": 3.5, "sims": 4, "source": "replay"},
        {"value_red": -0.5, "sims": 4, "source": "replay"},
    ]}
    evals = normalize_evals(rec)
    assert evals[0] is None
    assert evals[1] is None
    assert evals[2]["value_red"] == -0.5
