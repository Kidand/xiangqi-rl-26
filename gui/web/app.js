/**
 * app.js — 中国象棋 GUI 应用逻辑
 * 依赖：board.js（需先加载）
 * API 契约见 DESIGN.md §13
 */

"use strict";

// ═══════════════════════════════════════════════════════════════
//  常量与工具
// ═══════════════════════════════════════════════════════════════
// 同源相对路径：支持 --host 0.0.0.0 后手机/局域网直接访问（DESIGN §13）
const API_BASE = "";
const START_FEN = "rnbakabnr/9/1c5c1/p1p1p1p1p/9/9/P1P1P1P1P/1C5C1/9/RNBAKABNR w - - 0 1";

// ── Toast 通知 ──────────────────────────────────────────────────
function showToast(msg, isError = false, duration = 3000) {
  const ctr = document.getElementById("toast-container");
  const t = document.createElement("div");
  t.className = "toast" + (isError ? " error" : "");
  t.textContent = msg;
  ctr.appendChild(t);
  requestAnimationFrame(() => { requestAnimationFrame(() => { t.classList.add("show"); }); });
  setTimeout(() => {
    t.classList.remove("show");
    setTimeout(() => t.remove(), 300);
  }, duration);
}

// ── Fetch 封装（带错误处理） ─────────────────────────────────────
async function apiFetch(path, opts = {}) {
  const url = API_BASE + path;
  let res;
  try {
    res = await fetch(url, {
      headers: { "Content-Type": "application/json" },
      ...opts,
      body: opts.body !== undefined ? JSON.stringify(opts.body) : undefined
    });
  } catch (e) {
    showToast("网络错误：无法连接服务器", true);
    throw e;
  }
  const data = await res.json().catch(() => ({}));
  if (!res.ok) {
    const msg = data.error || data.detail || `HTTP ${res.status}`;
    showToast("错误：" + msg, true);
    throw new Error(msg);
  }
  return data;
}

// ═══════════════════════════════════════════════════════════════
//  WebAudio 音效
// ═══════════════════════════════════════════════════════════════
const AudioCtx = window.AudioContext || window.webkitAudioContext;
let _audioCtx = null;

function getAudioCtx() {
  if (!_audioCtx) _audioCtx = new AudioCtx();
  // 浏览器策略：非用户手势创建的 AudioContext 初始为 suspended，需显式 resume。
  if (_audioCtx.state === "suspended") _audioCtx.resume();
  return _audioCtx;
}

/** 简单音效：频率 + 时长（秒） */
function playBeep(freq, dur, type = "triangle", gainVal = 0.12) {
  try {
    const ctx = getAudioCtx();
    const osc = ctx.createOscillator();
    const gain = ctx.createGain();
    osc.connect(gain);
    gain.connect(ctx.destination);
    osc.type = type;
    osc.frequency.value = freq;
    gain.gain.setValueAtTime(gainVal, ctx.currentTime);
    gain.gain.exponentialRampToValueAtTime(0.001, ctx.currentTime + dur);
    osc.start(ctx.currentTime);
    osc.stop(ctx.currentTime + dur);
  } catch (_) { /* 浏览器静音策略不报错 */ }
}

function playMoveSound()    { playBeep(660, 0.08, "sine", 0.1); }
function playCaptureSound() { playBeep(330, 0.15, "triangle", 0.15); }

/** 统计 FEN 棋盘段的棋子数（字母=棋子，数字=空格，'/'=行分隔）。 */
function countFenPieces(fen) {
  if (!fen) return 0;
  const boardPart = String(fen).trim().split(/\s+/)[0];
  let n = 0;
  for (const ch of boardPart) {
    if ((ch >= "a" && ch <= "z") || (ch >= "A" && ch <= "Z")) n++;
  }
  return n;
}

// ═══════════════════════════════════════════════════════════════
//  应用状态
// ═══════════════════════════════════════════════════════════════
const App = {
  // 当前对局
  gameId: null,
  gameState: null,       // 最新 GameState
  mode: "hvh",           // "hva"|"ava"|"hvh"
  humanSide: "red",
  sims: 400,

  // 设置
  soundEnabled: true,
  autoFlipEnabled: false,

  // 模型是否已加载（用于 hvh 模式后台请求 with_eval 刷新评估条）
  modelLoaded: false,
  // 上一次渲染局面的棋子数（用于按 FEN 棋子数差判定吃子音效）
  lastPieceCount: null,

  // 复盘
  replayFens: [],
  replayMoves: [],
  replayChinese: [],
  replayResult: null,
  replayPos: 0,
  replayPlaying: false,
  replayTimer: null,
  replaySpeedMs: 1000,

  // 复盘胜率（DESIGN §12 evals：与 replayFens 等长，项为 null 或 {value_red,sims,source}）
  replayEvals: [],
  replayPath: null,        // records/ 内相对路径；文件/粘贴导入为 null（无法补算回写）
  replayIndex: 0,
  replayEvalToken: 0,      // 补算循环守卫：切换棋谱/开新对局时自增使旧循环退出
  replayEvalBusy: false,
  replayEvalPending: 0,

  // AVA 自动推进
  avaPlaying: false,
  avaTimer: null,

  // hva 两段式：AI 回着请求防重入
  aiReplyInFlight: false,

  // 评估条归属 = 棋盘当前显示的内容："game"（对局局面）| "replay"（复盘局面）。
  // 谁最后渲染棋盘谁拥有评估条；页签激活与否与归属无关（桌面布局棋盘常驻可见）。
  boardView: "game",

  // 提示
  hintVisible: false,

  // 历史局面跳转（对局中）
  viewingHistory: false,
  historyPos: -1,        // -1 = 最新
};

// ═══════════════════════════════════════════════════════════════
//  DOM 引用
// ═══════════════════════════════════════════════════════════════
let boardObj = null;   // XiangqiBoard 实例

function $id(id) { return document.getElementById(id); }

// ═══════════════════════════════════════════════════════════════
//  棋盘初始化
// ═══════════════════════════════════════════════════════════════
function initBoard() {
  // cellSize 固定 52（内部坐标系不变），SVG 按 viewBox 由 CSS 等比缩放，
  // 窗口尺寸变化无需重建棋盘（点击换算按 getBoundingClientRect 缩放，见 board.js）。
  const svgEl = $id("board-svg");
  boardObj = new XiangqiBoard(svgEl, { onMove: handleBoardMove });
  boardObj.setFen(START_FEN);
}

// ═══════════════════════════════════════════════════════════════
//  GameState 应用
// ═══════════════════════════════════════════════════════════════
function applyGameState(gs, withSound = true, opts = {}) {
  App.gameState = gs;
  App.viewingHistory = false;
  App.historyPos = -1;
  // 棋盘即将渲染对局局面 → 评估条归对局侧（见 boardView 约定）
  App.boardView = "game";

  boardObj.setFen(gs.fen);
  boardObj.setLegalMoves(gs.legal_moves || []);
  boardObj.setLastMove(gs.last_move || null);

  // 将军
  if (gs.in_check) {
    boardObj.setInCheck(gs.side_to_move);
  } else {
    boardObj.setInCheck(null);
  }

  // 音效：用前后两个局面的棋子数差判定吃子（notation 不含"吃"字，无法从中文判断）。
  // 仅在"走了一步"（withSound=true 且有 last_move）时触发；悔棋/返回/复盘等以
  // withSound=false 调用，只更新基线不发声，避免误播。
  const newPieceCount = countFenPieces(gs.fen);
  if (withSound && App.soundEnabled && gs.last_move) {
    if (App.lastPieceCount !== null && newPieceCount < App.lastPieceCount) {
      playCaptureSound();
    } else {
      // 基线未知（如 hva 执黑时新局响应已含 AI 首着）至少播走子音，
      // 不会误播吃子音（吃子判定要求基线已知）。
      playMoveSound();
    }
  }
  App.lastPieceCount = newPieceCount;

  // 评估条跟随棋盘内容（本函数已把 boardView 置为 "game"，无条件接管）。
  // 早期版本按「页签是否激活」判断归属是个 bug：桌面布局棋盘/评估条始终可见，
  // 右侧面板停在复盘页签时继续下棋会导致胜率条冻结。
  // opts.keepEvalBar：两段式走子中人类落子响应无 eval，保持上次评估不回弹 50%。
  if (gs.eval) updateEvalBar(gs.eval);
  else if (!opts.keepEvalBar) updateEvalBar(null);

  // 着法列表
  renderMoveList(gs.move_history || []);

  // 状态
  updateStatusFromGs(gs);

  // 交互开关
  const isMyTurn = isHumanTurn(gs);
  boardObj.setInteractive(isMyTurn && !gs.game_over);

  // 若对局结束
  if (gs.game_over) {
    boardObj.setInteractive(false);
    onGameOver(gs);
  }

  // 清除提示
  boardObj.setHint(null, []);
  App.hintVisible = false;
  $id("hint-area").classList.remove("visible");

  // 按钮显隐随状态更新（返回按钮 / AVA 播放按钮）
  updateModeButtons();

  // hvh 模式且已加载模型：后台异步请求 with_eval 刷新评估条（不阻塞交互）。
  // hva/ava 的 eval 已随 AI 落子附带，无需轮询。
  maybeRefreshEval();
}

/**
 * 按 App 状态显隐操作栏按钮（取代早期 index.html 内联轮询补丁）：
 * - btn-return-live：仅在查看历史局面时显示；
 * - btn-ava-toggle：仅在 AI 观战（ava）模式显示。
 * 状态变化点（applyGameState / startNewGame / returnToLive / jumpToHistoryMove）
 * 均显式调用本函数，无任何轮询。
 */
function updateModeButtons() {
  $id("btn-return-live").style.display = App.viewingHistory ? "" : "none";
  $id("btn-ava-toggle").style.display  = App.mode === "ava"  ? "" : "none";
}

function ucciToSq(s) {
  const col = "abcdefghi".indexOf(s[0].toLowerCase());
  const row  = parseInt(s[1], 10);
  return row * 9 + col;
}

function isHumanTurn(gs) {
  if (!gs || gs.game_over) return false;
  if (App.mode === "hvh") return true;
  if (App.mode === "ava") return false;
  // hva
  return gs.side_to_move === App.humanSide;
}

function updateStatusFromGs(gs) {
  if (!gs) { setStatus("就绪"); return; }
  if (gs.game_over) {
    const resultMap = { "1-0": "红方胜", "0-1": "黑方胜", "1/2-1/2": "和棋" };
    const termMap = {
      checkmate: "将死", stalemate: "困毙", repetition: "重复局面",
      perpetual_check: "长将判负", no_capture: "60回合无吃子", max_moves: "超长局"
    };
    const r = resultMap[gs.result] || gs.result;
    const t = gs.termination ? `（${termMap[gs.termination] || gs.termination}）` : "";
    setStatus(`对局结束：${r}${t}`);
  } else {
    const side = gs.side_to_move === "red" ? "红方" : "黑方";
    const chk  = gs.in_check ? "  ⚠ 将军！" : "";
    setStatus(`${side}走棋${chk}`);
  }
}

function setStatus(text) {
  $id("status-text").textContent = text;
}

function setThinking(flag) {
  $id("spinner").classList.toggle("active", flag);
  if (flag) setStatus("AI 思考中…");
}

function onGameOver(gs) {
  const resultMap = { "1-0": "红方获胜！", "0-1": "黑方获胜！", "1/2-1/2": "平局" };
  const msg = resultMap[gs.result] || "对局结束";
  setTimeout(() => showToast(msg, false, 4000), 300);
}

// ═══════════════════════════════════════════════════════════════
//  评估条
// ═══════════════════════════════════════════════════════════════
function updateEvalBar(evalObj) {
  // 只写 CSS 变量 --eval-pct（DESIGN §14）：填充轴向与锚定方向由 CSS 按断点决定
  // （桌面纵向红在下 / 移动横向红在左），JS 不感知布局。
  const fill = $id("eval-fill");
  const v = evalObj ? evalObj.value : null;
  if (v === null || v === undefined) {
    fill.style.setProperty("--eval-pct", "50%");
    return;
  }
  // eval.side（"red"|"black"）显式声明 value 是哪一方视角（= MCTS 运行时的走子方）。
  // 统一换算到红方视角：side==="red" 取 value，否则取 -value。
  // （不再靠 side_to_move 反推，那是历史上出过符号颠倒 bug 的推断链。）
  const redPov = evalObj.side === "red" ? v : -v;
  // 映射到 0..1 百分比（红方占比）
  const pct = (redPov + 1) / 2 * 100;
  fill.style.setProperty("--eval-pct", Math.max(2, Math.min(98, pct)) + "%");
}

/**
 * hvh 模式且模型已加载时，后台请求 with_eval 刷新评估条。
 * 用原生 fetch（而非 apiFetch）以避免后台失败时弹出错误 toast。
 * 请求返回后校验仍是同一局、非历史查看态才更新，避免竞态覆盖。
 */
async function maybeRefreshEval() {
  if (App.mode !== "hvh" || !App.modelLoaded || !App.gameId) return;
  const gs = App.gameState;
  if (!gs || gs.game_over || App.viewingHistory) return;
  const gid = App.gameId;
  // 评估条刷新是后台装饰功能，但服务端计算期间持有本局锁，
  // 会让紧随其后的 /move、/undo 排队——sims 封顶以限制持锁时长。
  const evalSims = Math.min(App.sims, 400);
  try {
    const res = await fetch(
      `${API_BASE}/api/game/${gid}?with_eval=true&sims=${evalSims}`,
      { headers: { "Content-Type": "application/json" } }
    );
    if (!res.ok) return;
    const data = await res.json();
    // 竞态守卫：除同局/非历史态外，还要求响应局面与当前局面一致，
    // 防止连走两步时先发请求的响应后到、用滞后一步的评估覆盖新值。
    const cur = App.gameState;
    if (App.gameId === gid && !App.viewingHistory &&
        App.boardView === "game" &&
        cur && !cur.game_over && data.fen === cur.fen) {
      updateEvalBar(data.eval || null);
    }
  } catch (_) { /* 后台刷新失败静默 */ }
}

// ═══════════════════════════════════════════════════════════════
//  着法列表（对局）
// ═══════════════════════════════════════════════════════════════
function renderMoveList(history) {
  const list = $id("moves-list");
  list.innerHTML = "";

  if (!history.length) {
    const empty = document.createElement("div");
    empty.className = "text-sm text-muted";
    empty.style.padding = "8px 10px";
    empty.textContent = "暂无着法";
    list.appendChild(empty);
    return;
  }

  for (let i = 0; i < history.length; i++) {
    const entry = history[i];
    const moveNum = Math.floor(i / 2) + 1;
    const isRed   = (i % 2 === 0);   // 回合始终红先

    const row = document.createElement("div");
    row.className = "move-row";

    if (isRed) {
      const idx = document.createElement("span");
      idx.className = "move-idx";
      idx.textContent = moveNum + ".";
      row.appendChild(idx);
    } else {
      const spacer = document.createElement("span");
      spacer.className = "move-idx";
      row.appendChild(spacer);
    }

    const item = document.createElement("span");
    item.className = "move-item " + (isRed ? "red-move" : "black-move");
    item.textContent = entry.chinese || entry.move;
    item.dataset.idx = String(i);
    item.addEventListener("click", () => jumpToHistoryMove(i, entry.fen_after));
    row.appendChild(item);

    // 若为最后一步，标记 current
    if (i === history.length - 1 && !App.viewingHistory) {
      item.classList.add("current");
    }

    list.appendChild(row);
  }

  // 自动滚到底部
  list.scrollTop = list.scrollHeight;
}

function jumpToHistoryMove(idx, fenAfter) {
  if (!fenAfter) return;
  App.viewingHistory = true;
  App.historyPos = idx;

  // 渲染历史局面（仅显示，不可走子）
  const history = App.gameState ? App.gameState.move_history : [];
  boardObj.setFen(fenAfter);
  boardObj.setLegalMoves([]);
  boardObj.setInteractive(false);

  const lastMv = history[idx] ? history[idx].move : null;
  boardObj.setLastMove(lastMv);
  boardObj.setInCheck(null);
  boardObj.setHint(null, []);

  // 高亮当前步
  const items = $id("moves-list").querySelectorAll(".move-item");
  items.forEach(it => it.classList.remove("current"));
  const cur = $id("moves-list").querySelector(`[data-idx="${idx}"]`);
  if (cur) { cur.classList.add("current"); cur.scrollIntoView({ block: "nearest" }); }

  setStatus(`查看历史 第${idx+1}步`);
  updateModeButtons();   // 显示「↩ 返回」按钮
}

function returnToLive() {
  if (!App.gameState) return;
  App.viewingHistory = false;
  App.historyPos = -1;
  applyGameState(App.gameState, false);
  updateModeButtons();
}

// ═══════════════════════════════════════════════════════════════
//  新对局对话框
// ═══════════════════════════════════════════════════════════════
function openNewGameDialog() {
  $id("dialog-new-game").classList.add("open");
}

function closeNewGameDialog() {
  $id("dialog-new-game").classList.remove("open");
}

async function startNewGame() {
  const mode       = getRadioValue("new-mode");
  const humanSide  = getRadioValue("new-side");
  const simsVal    = getRadioValue("new-sims");
  const customFen  = $id("new-fen").value.trim();
  const sims       = parseInt(simsVal, 10);

  App.mode = mode;
  App.humanSide = humanSide;
  App.sims = sims;

  closeNewGameDialog();
  stopAva();
  stopReplay();
  // 清空复盘数据，防止方向键仍然导航旧复盘；token 自增终止后台补算循环
  App.replayEvalToken++;
  App.replayEvalBusy = false;
  App.replayEvalPending = 0;
  App.replayFens = [];
  App.replayMoves = [];
  App.replayChinese = [];
  App.replayEvals = [];
  App.replayPath = null;
  $id("replay-eval-text").textContent = "—";
  // 重置吃子音效基线：新局第一次 applyGameState 不与上一局棋子数比较，避免误播
  App.lastPieceCount = null;

  const body = { mode, human_side: humanSide, sims };
  if (customFen) body.fen = customFen;

  // 两段式（DESIGN §13/§14）：新局立即返回渲染，AI 首着（若有）随后单独请求
  body.ai_reply = false;

  setThinking(true);
  try {
    const gs = await apiFetch("/api/game/new", { method: "POST", body });
    App.gameId = gs.game_id;
    applyGameState(gs);
    updateModeButtons();

    // AVA 模式：开始自动推进
    if (mode === "ava") {
      $id("tab-game-btn").click();
      scheduleAva();
    } else if (mode === "hva" && !gs.game_over && gs.side_to_move !== humanSide) {
      // AI 执红先行：棋盘已上屏，再取 AI 首着
      await requestAiReply();
    }
  } catch (_) {
    // 错误已由 apiFetch 处理
  } finally {
    setThinking(false);
  }
}

function getRadioValue(name) {
  const el = document.querySelector(`input[name="${name}"]:checked`);
  return el ? el.value : null;
}

// ═══════════════════════════════════════════════════════════════
//  走子处理
// ═══════════════════════════════════════════════════════════════
async function handleBoardMove(fromSq, toSq) {
  if (!App.gameId) return;
  if (App.viewingHistory) { returnToLive(); return; }

  const col1 = fromSq % 9, row1 = Math.floor(fromSq / 9);
  const col2 = toSq   % 9, row2 = Math.floor(toSq   / 9);
  const move = "abcdefghi"[col1] + row1 + "abcdefghi"[col2] + row2;

  boardObj.setInteractive(false);
  setThinking(true);
  let gs;
  try {
    // 两段式（DESIGN §13/§14）：ai_reply=false 只落人类子立即返回
    gs = await apiFetch(`/api/game/${App.gameId}/move`, {
      method: "POST",
      body: { move, ai_reply: false }
    });
  } catch (_) {
    // 恢复交互
    if (App.gameState) {
      boardObj.setInteractive(isHumanTurn(App.gameState));
    }
    setThinking(false);
    return;
  }

  // 玩家落子立即上屏；AI 思考期间棋盘已是新局面
  applyGameState(gs, true, { keepEvalBar: true });

  if (App.mode === "hva" && !gs.game_over) {
    await requestAiReply();
  } else {
    setThinking(false);
  }
}

/**
 * hva 两段式：请求 AI 回着（玩家落子已先行渲染）。
 * - 网络/5xx 等瞬时错误重试一次；
 * - 400/409 是终态（终局/悔棋竞态被轮次守卫拦下），不重试、不覆盖状态栏，
 *   改为拉取服务器最新局面同步显示；
 * - gameId 变化（开了新局）随时中止；in-flight 防重入。
 */
async function requestAiReply() {
  const gid = App.gameId;
  if (!gid || App.aiReplyInFlight) return;
  App.aiReplyInFlight = true;
  setThinking(true);
  try {
    for (let attempt = 0; attempt < 2; attempt++) {
      if (App.gameId !== gid) return;
      let res;
      try {
        res = await fetch(`${API_BASE}/api/game/${gid}/ai_move`, {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ sims: App.sims }),
        });
      } catch (_) {
        if (attempt === 0) { await new Promise(r => setTimeout(r, 800)); continue; }
        showToast("网络错误：AI 走子失败", true);
        if (App.gameId === gid) setStatus("AI 走子失败：点「悔棋」撤回后重走");
        return;
      }
      if (App.gameId !== gid) return;
      const data = await res.json().catch(() => ({}));
      if (res.ok) {
        applyGameState(data);
        return;
      }
      if (res.status === 400 || res.status === 409) {
        // 终态拒绝：undo/resign 已抢先改变局面，以服务器状态为准，不打扰用户
        await refreshGameState(gid);
        return;
      }
      if (attempt === 0) { await new Promise(r => setTimeout(r, 800)); continue; }
      showToast("错误：" + (data.error || `HTTP ${res.status}`), true);
      if (App.gameId === gid) setStatus("AI 走子失败：点「悔棋」撤回后重走");
      return;
    }
  } finally {
    App.aiReplyInFlight = false;
    if (App.gameId === gid) setThinking(false);
  }
}

/** 以服务器为准同步一次对局状态（终态 4xx 后恢复一致显示）。 */
async function refreshGameState(gid) {
  try {
    const res = await fetch(`${API_BASE}/api/game/${gid}`, {
      headers: { "Content-Type": "application/json" },
    });
    if (!res.ok) return;
    const gs = await res.json();
    if (App.gameId === gid) applyGameState(gs, false, { keepEvalBar: true });
  } catch (_) { /* 同步失败静默，界面维持现状 */ }
}

// ═══════════════════════════════════════════════════════════════
//  AVA 自动推进
// ═══════════════════════════════════════════════════════════════
function scheduleAva() {
  if (App.avaPlaying) return;
  App.avaPlaying = true;
  updateAvaBtn();
  avaStep();
}

function stopAva() {
  App.avaPlaying = false;
  if (App.avaTimer) { clearTimeout(App.avaTimer); App.avaTimer = null; }
  updateAvaBtn();
}

function toggleAva() {
  if (App.avaPlaying) stopAva();
  else scheduleAva();
}

function updateAvaBtn() {
  const btn = $id("btn-ava-toggle");
  if (!btn) return;
  btn.textContent = App.avaPlaying ? "⏸ 暂停" : "▶ 继续";
}

async function avaStep() {
  if (!App.avaPlaying) return;
  if (!App.gameId) return;
  if (App.gameState && App.gameState.game_over) { stopAva(); return; }
  if (App.mode !== "ava") { stopAva(); return; }

  setThinking(true);
  try {
    const gs = await apiFetch(`/api/game/${App.gameId}/ai_move`, {
      method: "POST",
      body: { sims: App.sims }
    });
    applyGameState(gs);
    if (gs.game_over) { stopAva(); return; }
  } catch (_) {
    stopAva();
    return;
  } finally {
    setThinking(false);
  }

  // 延迟下一步（给用户看的时间）
  App.avaTimer = setTimeout(() => { if (App.avaPlaying) avaStep(); }, 600);
}

// ═══════════════════════════════════════════════════════════════
//  功能按钮
// ═══════════════════════════════════════════════════════════════
async function doUndo() {
  if (!App.gameId) return;
  if (App.viewingHistory) { returnToLive(); return; }
  try {
    const gs = await apiFetch(`/api/game/${App.gameId}/undo`, { method: "POST" });
    applyGameState(gs, false);
  } catch (_) {}
  // hva 兜底：若撤后（或撤失败但本就）轮到 AI 且未终局，重新取 AI 着，
  // 覆盖「AI 回着失败后悔棋」「人执黑撤掉 AI 首着」等恢复路径。
  const cur = App.gameState;
  if (App.mode === "hva" && cur && !cur.game_over &&
      cur.side_to_move !== App.humanSide) {
    await requestAiReply();
  }
}

async function doResign() {
  if (!App.gameId) return;
  if (!confirm("确认认输？")) return;
  try {
    const gs = await apiFetch(`/api/game/${App.gameId}/resign`, { method: "POST" });
    applyGameState(gs, false);
  } catch (_) {}
}

function doFlip() {
  boardObj.flip();
}

async function doExport() {
  if (!App.gameId) return;
  try {
    const data = await apiFetch(`/api/game/${App.gameId}/export`);
    const blob = new Blob([JSON.stringify(data, null, 2)], { type: "application/json" });
    const url  = URL.createObjectURL(blob);
    const a    = document.createElement("a");
    a.href     = url;
    a.download = `xiangqi_${App.gameId}_${new Date().toISOString().slice(0,19).replace(/:/g,"-")}.json`;
    a.click();
    URL.revokeObjectURL(url);
  } catch (_) {}
}

async function doHint() {
  if (!App.gameId) return;
  if (App.viewingHistory) return;
  setThinking(true);
  try {
    const data = await apiFetch(`/api/game/${App.gameId}/hint?sims=${App.sims}`);
    boardObj.setHint(data.move, data.top_moves || []);
    renderHintArea(data);
    App.hintVisible = true;
    $id("hint-area").classList.add("visible");
  } catch (_) {}
  finally { setThinking(false); }
}

function renderHintArea(data) {
  const list = $id("hint-moves");
  list.innerHTML = "";
  const tops = data.top_moves || [];
  if (!tops.length && data.move) {
    tops.push({ move: data.move, chinese: data.chinese, prob: null, visits: null });
  }
  tops.forEach((tm, i) => {
    const row = document.createElement("div");
    row.className = "hint-move-item";

    const rank = document.createElement("span");
    rank.className = "hint-rank";
    rank.textContent = i + 1;

    const cn = document.createElement("span");
    cn.className = "hint-cn";
    cn.textContent = tm.chinese || tm.move;

    const prob = document.createElement("span");
    prob.className = "hint-prob";
    prob.textContent = tm.prob != null ? (tm.prob * 100).toFixed(1) + "%" : "";

    const visits = document.createElement("span");
    visits.className = "hint-visits";
    visits.textContent = tm.visits != null ? `${tm.visits}次` : "";

    row.appendChild(rank);
    row.appendChild(cn);
    row.appendChild(prob);
    row.appendChild(visits);

    // 点击上箭头
    row.addEventListener("click", () => {
      boardObj.setHint(tm.move, tops);
    });
    list.appendChild(row);
  });
}

// ═══════════════════════════════════════════════════════════════
//  复盘功能
// ═══════════════════════════════════════════════════════════════

// ── 文件导入 ─────────────────────────────────────────────────
function setupReplayImport() {
  $id("btn-replay-file").addEventListener("click", () => {
    $id("replay-file-input").click();
  });

  $id("replay-file-input").addEventListener("change", async (e) => {
    const file = e.target.files[0];
    if (!file) return;
    const text = await file.text();
    e.target.value = "";
    await loadReplayFromText(text, file.name);
  });

  $id("btn-replay-paste").addEventListener("click", () => {
    const pa = $id("replay-paste-area");
    const btns = $id("replay-paste-btns");
    const visible = pa.classList.toggle("visible");
    btns.classList.toggle("visible", visible);
  });

  $id("btn-paste-load").addEventListener("click", async () => {
    const text = $id("replay-paste-area").value.trim();
    if (!text) return;
    await loadReplayFromText(text, "粘贴");
    $id("replay-paste-area").classList.remove("visible");
    $id("replay-paste-btns").classList.remove("visible");
  });

  $id("btn-paste-clear").addEventListener("click", () => {
    $id("replay-paste-area").value = "";
    $id("replay-paste-area").classList.remove("visible");
    $id("replay-paste-btns").classList.remove("visible");
  });
}

async function loadReplayFromText(text, filename) {
  text = text.trim();
  let body;

  // 尝试解析为 JSON
  try {
    const parsed = JSON.parse(text);
    // 可能是单局 v1 JSON 或 JSONL（取第一局）
    if (parsed.format === "xiangqi-record-v1" || parsed.moves) {
      body = { record: parsed };
    } else {
      showToast("JSON 格式不识别", true);
      return;
    }
  } catch (_) {
    // JSONL：取首条
    const firstLine = text.split("\n").find(l => l.trim());
    try {
      const parsed = JSON.parse(firstLine);
      body = { record: parsed };
    } catch (_) {
      // 纯文本格式
      body = { text };
    }
  }

  try {
    const data = await apiFetch("/api/replay/load", { method: "POST", body });
    loadReplayData(data);
    showToast(`已加载：${filename}（${data.moves.length} 步）`);
    $id("tab-replay-btn").click();
  } catch (_) {}
}

function loadReplayData(data, srcPath = null, srcIndex = 0) {
  stopReplay();
  App.replayEvalToken++;           // 终止上一份棋谱的补算循环
  App.replayEvalBusy = false;
  App.replayEvalPending = 0;
  App.replayFens    = data.fens || [];
  App.replayMoves   = data.moves || [];
  App.replayChinese = data.chinese || [];
  App.replayResult  = data.result || null;
  App.replayPos     = 0;
  App.replayEvals   = (Array.isArray(data.evals) && data.evals.length === App.replayFens.length)
                        ? data.evals.slice()
                        : new Array(App.replayFens.length).fill(null);
  App.replayPath    = srcPath;
  App.replayIndex   = srcIndex;

  // 显示信息
  const info = $id("replay-info");
  const total = App.replayMoves.length;
  const resMap = { "1-0": "红方胜", "0-1": "黑方胜", "1/2-1/2": "和棋" };
  info.textContent = `共 ${total} 步 | 结果：${resMap[App.replayResult] || App.replayResult || "未知"}`;

  renderReplayMoveList();
  gotoReplayPos(0);
  maybeStartReplayEvalLoop();
}

function renderReplayMoveList() {
  const list = $id("replay-moves-list");
  list.innerHTML = "";

  const history = App.replayMoves;
  for (let i = 0; i < history.length; i++) {
    const moveNum = Math.floor(i / 2) + 1;
    const isRed   = (i % 2 === 0);

    const row = document.createElement("div");
    row.className = "move-row";

    if (isRed) {
      const idx = document.createElement("span");
      idx.className = "move-idx";
      idx.textContent = moveNum + ".";
      row.appendChild(idx);
    } else {
      const spacer = document.createElement("span");
      spacer.className = "move-idx";
      row.appendChild(spacer);
    }

    const item = document.createElement("span");
    item.className = "move-item " + (isRed ? "red-move" : "black-move");
    item.textContent = App.replayChinese[i] || history[i];
    item.dataset.idx = String(i + 1);  // pos i+1 代表执行第 i 步之后
    item.addEventListener("click", () => gotoReplayPos(i + 1));
    row.appendChild(item);

    list.appendChild(row);
  }
}

function gotoReplayPos(pos) {
  App.replayPos = Math.max(0, Math.min(App.replayFens.length - 1, pos));
  const fen = App.replayFens[App.replayPos];
  if (!fen) return;

  // 棋盘即将渲染复盘局面 → 评估条归复盘侧
  App.boardView = "replay";

  boardObj.setFen(fen);
  boardObj.setLegalMoves([]);
  boardObj.setInteractive(false);
  boardObj.setInCheck(null);
  boardObj.setHint(null, []);

  // 高亮最后一步
  if (App.replayPos > 0 && App.replayMoves[App.replayPos - 1]) {
    boardObj.setLastMove(App.replayMoves[App.replayPos - 1]);
  } else {
    boardObj.setLastMove(null);
  }

  // 更新位置标签
  $id("replay-pos-label").textContent =
    `${App.replayPos} / ${App.replayFens.length - 1}`;

  // 高亮着法列表
  const items = $id("replay-moves-list").querySelectorAll(".move-item");
  items.forEach(it => it.classList.remove("current"));
  if (App.replayPos > 0) {
    const cur = $id("replay-moves-list").querySelector(`[data-idx="${App.replayPos}"]`);
    if (cur) { cur.classList.add("current"); cur.scrollIntoView({ block: "nearest" }); }
  }

  // 当前局面的红黑胜率
  updateReplayEvalDisplay();
}

// ── 复盘胜率显示与后台补算 ──────────────────────────────────
function updateReplayEvalDisplay() {
  const el = $id("replay-eval-text");
  if (!App.replayFens.length) { el.textContent = "—"; return; }

  // 评估条只在棋盘正在显示复盘局面时接管（避免覆盖对局的实时评估）
  const replayActive = App.boardView === "replay";
  const ev = App.replayEvals[App.replayPos] || null;
  let txt;
  if (ev && typeof ev.value_red === "number") {
    // value_red 恒为红方视角（DESIGN §12），以 side:"red" 复用评估条换算
    if (replayActive) updateEvalBar({ value: ev.value_red, side: "red" });
    const red = (ev.value_red + 1) / 2 * 100;
    const srcMap = { play: "对局实时", replay: "复盘补算", result: "终局" };
    txt = `红 ${red.toFixed(1)}% ｜ 黑 ${(100 - red).toFixed(1)}%`
        + `（${srcMap[ev.source] || "已计算"}${ev.sims ? `·${ev.sims}模拟` : ""}）`;
  } else {
    if (replayActive) updateEvalBar(null);
    if (App.replayEvalBusy) txt = "该局面胜率计算中…";
    else if (!App.replayPath) txt = "导入棋谱无此局面胜率（从自对弈记录打开可自动补算）";
    else if (!App.modelLoaded) txt = "未加载模型，无法计算胜率";
    else txt = "该局面胜率未计算";
  }
  if (App.replayEvalBusy && App.replayEvalPending > 0) {
    txt += ` ｜ 后台补算中，剩余 ${App.replayEvalPending} 个局面`;
  }
  el.textContent = txt;
}

/**
 * records/ 打开的棋谱若 evals 有缺失且模型已加载，后台循环调
 * /api/records/eval 分批补算（服务端每批算完即回写文件），随算随刷新。
 * token 守卫：切换棋谱/开新对局时 App.replayEvalToken 自增，旧循环退出。
 */
async function maybeStartReplayEvalLoop() {
  if (!App.replayPath || !App.modelLoaded) return;
  if (!App.replayEvals.some(e => !e)) return;

  const token = ++App.replayEvalToken;
  App.replayEvalBusy = true;
  App.replayEvalPending = App.replayEvals.filter(e => !e).length;
  updateReplayEvalDisplay();
  try {
    while (true) {
      const res = await apiFetch("/api/records/eval", {
        method: "POST",
        body: { path: App.replayPath, index: App.replayIndex, sims: 128, max_positions: 8 },
      });
      if (token !== App.replayEvalToken) return;   // 已切换棋谱，结果作废
      if (Array.isArray(res.evals) && res.evals.length === App.replayFens.length) {
        App.replayEvals = res.evals;
      }
      App.replayEvalPending = res.pending || 0;
      updateReplayEvalDisplay();
      if (!res.pending) break;
    }
  } catch (_) {
    /* apiFetch 已弹 toast；停止本轮补算 */
  } finally {
    if (token === App.replayEvalToken) {
      App.replayEvalBusy = false;
      App.replayEvalPending = 0;
      updateReplayEvalDisplay();
    }
  }
}

// ── 播放控制 ─────────────────────────────────────────────────
function setupReplayControls() {
  $id("ctrl-first").addEventListener("click",  () => { stopReplay(); gotoReplayPos(0); });
  $id("ctrl-prev").addEventListener("click",   () => { stopReplay(); gotoReplayPos(App.replayPos - 1); });
  $id("ctrl-next").addEventListener("click",   () => { stopReplay(); gotoReplayPos(App.replayPos + 1); });
  $id("ctrl-last").addEventListener("click",   () => { stopReplay(); gotoReplayPos(App.replayFens.length - 1); });
  $id("ctrl-play").addEventListener("click",   () => toggleReplay());

  $id("replay-speed").addEventListener("input", (e) => {
    // 滑块值 1..5，映射到 ms
    const speeds = [2000, 1200, 800, 500, 300];
    App.replaySpeedMs = speeds[parseInt(e.target.value, 10) - 1] || 1000;
  });
}

function toggleReplay() {
  if (App.replayPlaying) stopReplay();
  else startReplay();
}

function startReplay() {
  if (!App.replayFens.length) return;
  App.replayPlaying = true;
  $id("ctrl-play").classList.add("active");
  $id("ctrl-play").textContent = "⏸";
  replayStep();
}

function stopReplay() {
  App.replayPlaying = false;
  if (App.replayTimer) { clearTimeout(App.replayTimer); App.replayTimer = null; }
  const btn = $id("ctrl-play");
  if (btn) { btn.classList.remove("active"); btn.textContent = "▶"; }
}

function replayStep() {
  if (!App.replayPlaying) return;
  if (App.replayPos >= App.replayFens.length - 1) {
    stopReplay();
    return;
  }
  gotoReplayPos(App.replayPos + 1);
  App.replayTimer = setTimeout(replayStep, App.replaySpeedMs);
}

// ── records 浏览 ────────────────────────────────────────────
async function loadRecordsList() {
  const list = $id("records-list");
  list.innerHTML = '<div class="text-sm text-muted" style="padding:6px 10px">加载中…</div>';
  try {
    const data = await apiFetch("/api/records/list");
    const files = data.files || [];
    list.innerHTML = "";
    if (!files.length) {
      list.innerHTML = '<div class="text-sm text-muted" style="padding:6px 10px">暂无记录</div>';
      return;
    }
    files.forEach(f => {
      const item = document.createElement("div");
      item.className = "record-item";
      const name = document.createElement("span");
      name.className = "record-name";
      name.textContent = f.path.split("/").pop().split("\\").pop();
      name.title = f.path;
      const meta = document.createElement("span");
      meta.className = "record-meta";
      meta.textContent = `${f.games}局`;
      item.appendChild(name);
      item.appendChild(meta);
      item.addEventListener("click", () => openRecordFile(f, item));
      list.appendChild(item);
    });
  } catch (_) {
    list.innerHTML = '<div class="text-sm text-muted" style="padding:6px 10px">加载失败</div>';
  }
}

async function openRecordFile(fileInfo, itemEl) {
  // 高亮选中
  $id("records-list").querySelectorAll(".record-item").forEach(i => i.classList.remove("active"));
  itemEl.classList.add("active");

  const sel = $id("records-game-select");
  sel.innerHTML = "";
  sel.classList.add("visible");

  // 若只有 1 局直接加载
  if (fileInfo.games === 1) {
    sel.classList.remove("visible");
    await loadRecordGame(fileInfo.path, 0);
    return;
  }

  // 多局：显示下拉
  for (let i = 0; i < fileInfo.games; i++) {
    const opt = document.createElement("option");
    opt.value = String(i);
    opt.textContent = `第 ${i + 1} 局`;
    sel.appendChild(opt);
  }
  sel.onchange = async () => {
    await loadRecordGame(fileInfo.path, parseInt(sel.value, 10));
  };
  // 默认加载第 0 局
  await loadRecordGame(fileInfo.path, 0);
}

async function loadRecordGame(path, index) {
  try {
    const record = await apiFetch(`/api/records/get?path=${encodeURIComponent(path)}&index=${index}`);
    const data   = await apiFetch("/api/replay/load", { method: "POST", body: { record } });
    loadReplayData(data, path, index);   // 带上来源路径，缺失胜率可补算回写
    showToast(`已加载第 ${index + 1} 局`);
  } catch (_) {}
}

// ═══════════════════════════════════════════════════════════════
//  模型信息 & 设置
// ═══════════════════════════════════════════════════════════════
async function refreshModelInfo() {
  try {
    const info = await apiFetch("/api/model/info");
    App.modelLoaded = !!info.loaded;
    const box = $id("model-info-box");
    if (info.loaded) {
      box.innerHTML =
        `<b>已加载</b>：${info.path || "—"}<br>` +
        `参数：${info.params ? (info.params/1e6).toFixed(2) + "M" : "—"} | ` +
        `${info.blocks}块 × ${info.filters}滤波器<br>` +
        `设备：${info.device || "cpu"}`;
    } else {
      box.textContent = "未加载模型（hvh 模式不需要）";
    }
  } catch (_) {
    $id("model-info-box").textContent = "无法获取模型信息";
  }
}

async function loadModel() {
  const path = $id("model-path-input").value.trim();
  if (!path) return;
  try {
    const res = await apiFetch("/api/model/load", { method: "POST", body: { path } });
    if (res.ok) {
      App.modelLoaded = true;
      showToast("模型加载成功");
      await refreshModelInfo();
      // 若当前正处于 hvh 对局，立即刷新一次评估条
      maybeRefreshEval();
      // 复盘已打开且缺胜率：模型就绪后启动补算（并刷新提示文案）
      updateReplayEvalDisplay();
      maybeStartReplayEvalLoop();
    }
  } catch (_) {}
}

// ═══════════════════════════════════════════════════════════════
//  页签切换
// ═══════════════════════════════════════════════════════════════
function setupTabs() {
  const tabs = [
    { btn: "tab-game-btn",     pane: "tab-game" },
    { btn: "tab-replay-btn",   pane: "tab-replay" },
    { btn: "tab-settings-btn", pane: "tab-settings" },
  ];

  tabs.forEach(({ btn, pane }) => {
    $id(btn).addEventListener("click", () => {
      tabs.forEach(t => {
        $id(t.btn).classList.remove("active");
        $id(t.pane).classList.remove("active");
      });
      $id(btn).classList.add("active");
      $id(pane).classList.add("active");

      if (pane === "tab-replay") {
        loadRecordsList();
        // 已加载棋谱：重新渲染当前复盘局面（棋盘+评估条一起交给复盘侧）
        if (App.replayFens.length) gotoReplayPos(App.replayPos);
      } else if (pane === "tab-settings") {
        refreshModelInfo();
      } else if (pane === "tab-game") {
        // 回到对局页：若棋盘正显示复盘局面，恢复为对局局面（含交互与评估条）
        if (App.gameState && App.boardView === "replay") {
          applyGameState(App.gameState, false);
        } else {
          maybeRefreshEval();
        }
      }
    });
  });
}

// ═══════════════════════════════════════════════════════════════
//  设置开关
// ═══════════════════════════════════════════════════════════════
function setupSettings() {
  const soundToggle = $id("toggle-sound");
  soundToggle.checked = App.soundEnabled;
  soundToggle.addEventListener("change", () => { App.soundEnabled = soundToggle.checked; });

  const flipToggle = $id("toggle-auto-flip");
  flipToggle.checked = App.autoFlipEnabled;
  flipToggle.addEventListener("change", () => { App.autoFlipEnabled = flipToggle.checked; });

  $id("btn-load-model").addEventListener("click", loadModel);
}

// ═══════════════════════════════════════════════════════════════
//  键盘快捷键
// ═══════════════════════════════════════════════════════════════
function setupKeyboard() {
  document.addEventListener("keydown", (e) => {
    // 焦点在输入框时不处理
    if (e.target.tagName === "INPUT" || e.target.tagName === "TEXTAREA") return;

    switch (e.key) {
      case "ArrowLeft":
        e.preventDefault();
        if (App.replayFens.length > 0) { stopReplay(); gotoReplayPos(App.replayPos - 1); }
        else if (App.viewingHistory) jumpToHistoryMove(App.historyPos - 1, getHistoryFen(App.historyPos - 1));
        break;
      case "ArrowRight":
        e.preventDefault();
        if (App.replayFens.length > 0) { stopReplay(); gotoReplayPos(App.replayPos + 1); }
        else if (App.viewingHistory) jumpToHistoryMove(App.historyPos + 1, getHistoryFen(App.historyPos + 1));
        break;
      case "Home":
        e.preventDefault();
        if (App.replayFens.length > 0) { stopReplay(); gotoReplayPos(0); }
        break;
      case "End":
        e.preventDefault();
        if (App.replayFens.length > 0) { stopReplay(); gotoReplayPos(App.replayFens.length - 1); }
        break;
      case " ":
        e.preventDefault();
        if (App.replayFens.length > 0) toggleReplay();
        else if (App.mode === "ava") toggleAva();
        break;
      case "Escape":
        if ($id("dialog-new-game").classList.contains("open")) closeNewGameDialog();
        if (App.viewingHistory) returnToLive();
        break;
    }
  });
}

function getHistoryFen(idx) {
  if (!App.gameState) return null;
  const hist = App.gameState.move_history || [];
  if (idx < 0 || idx >= hist.length) return null;
  return hist[idx].fen_after;
}

// ═══════════════════════════════════════════════════════════════
//  初始化入口
// ═══════════════════════════════════════════════════════════════
document.addEventListener("DOMContentLoaded", () => {
  initBoard();
  setupTabs();
  setupSettings();
  setupReplayImport();
  setupReplayControls();
  setupKeyboard();

  // 操作按钮
  $id("btn-new-game").addEventListener("click", openNewGameDialog);
  $id("btn-undo").addEventListener("click", doUndo);
  $id("btn-resign").addEventListener("click", doResign);
  $id("btn-flip").addEventListener("click", doFlip);
  $id("btn-export").addEventListener("click", doExport);
  $id("btn-hint").addEventListener("click", doHint);

  // 对话框
  $id("dlg-start").addEventListener("click", startNewGame);
  $id("dlg-cancel").addEventListener("click", closeNewGameDialog);
  $id("dialog-new-game").addEventListener("click", (e) => {
    if (e.target === $id("dialog-new-game")) closeNewGameDialog();
  });

  // 对战模式单选：AI 观战（ava）不需要选边，隐藏执子方选项
  document.querySelectorAll('input[name="new-mode"]').forEach(r => {
    r.addEventListener("change", () => {
      $id("side-group").style.display = getRadioValue("new-mode") === "ava" ? "none" : "";
    });
  });

  // AVA 按钮
  const avaToggle = $id("btn-ava-toggle");
  if (avaToggle) avaToggle.addEventListener("click", toggleAva);

  // records 刷新按钮
  $id("btn-records-refresh").addEventListener("click", loadRecordsList);

  // 返回当前局面按钮
  $id("btn-return-live").addEventListener("click", returnToLive);

  // 初始状态
  updateModeButtons();
  setStatus("就绪 — 点击「新对局」开始");
  refreshModelInfo();
});
