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
def test_records_get(client: SimpleClient, tmp_path: Path):
    """在 records/ 下写入一个测试记录文件后通过 API 读取。"""
    import xiangqi.records as rmod
    from xiangqi.constants import START_FEN

    # 确保 records/ 目录存在
    repo_root = Path(__file__).resolve().parent.parent
    rec_dir = repo_root / "records"
    rec_dir.mkdir(exist_ok=True)

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
# 测试：hva eval.value 语义（评估条符号契约）
# DESIGN §13: eval.value 随 GameState 附带，联合 side_to_move 解读。
# 约定：AI 走子前 MCTS 以 AI 方视角运算，walk_val 为 AI 期望值；
#        AI 走子后 side_to_move 切到人类侧：
#          - AI=黑 → side_to_move="red"  → 红方 POV = -eval.value
#          - AI=红 → side_to_move="black" → 红方 POV = +eval.value
# 该契约决定前端 updateEvalBar 必须在 side_to_move=="red" 时取反（而非"black"）。
# ────────────────────────────────────────────────────────────────

def test_hva_eval_value_in_range_human_red(client_with_model: SimpleClient):
    """hva human=red：AI（黑）回着后 eval.value ∈ [-1,1]，side_to_move 应回到 "red"。"""
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

    # 约定：AI=黑走后 side_to_move="red"；前端应对 eval.value 取反得红方 POV
    # （见 app.js updateEvalBar：negate when side_to_move === "red"）
    # 此处只验证符号约定可推导：不测具体值，只确认字段存在且在范围内。
    assert ev.get("top_moves") is not None, "eval 缺少 top_moves 字段"


def test_hva_eval_value_in_range_human_black(client_with_model: SimpleClient):
    """hva human=black：AI（红）先走，eval.value ∈ [-1,1]，side_to_move 应为 "black"。"""
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

    # 约定：AI=红走后 side_to_move="black"；前端不应对 eval.value 取反（eval 已是红方视角）
    top = ev.get("top_moves", [])
    # top_moves 应按概率降序排列
    for i in range(len(top) - 1):
        assert top[i]["prob"] >= top[i+1]["prob"] - 1e-9, (
            f"top_moves 未降序：[{i}].prob={top[i]['prob']} > [{i+1}].prob={top[i+1]['prob']}"
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
