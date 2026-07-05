/**
 * board.js — 中国象棋 SVG 棋盘组件
 * 依赖：无（不依赖 app.js）
 * 坐标系：row 0 = 红方底线（下），row 9 = 黑方底线（上），col 0..8 = a..i
 * FEN 解析按 DESIGN.md §3
 */

"use strict";

// ── FEN 解析 ─────────────────────────────────────────────────────────────────
// 棋子字符 -> { side: 'red'|'black', type: 字母(大写标准) }
const CHAR_ALIAS = { E: "B", e: "b", H: "N", h: "n" };
const PIECE_CHARS_RED = new Set(["K","A","B","N","R","C","P"]);
const PIECE_CHARS_BLACK = new Set(["k","a","b","n","r","c","p"]);

/**
 * 解析 FEN 棋盘段，返回 90 元素数组。
 * board[sq] = null | { side:'red'|'black', type:'K'|'A'|'B'|'N'|'R'|'C'|'P' }
 * sq = row*9+col，row 0=红方底线（下），row 9=黑方底线（上）
 * FEN 棋盘段从 row 9 开始向下到 row 0，每行 col 0..8
 */
function parseFenBoard(fenBoard) {
  const board = new Array(90).fill(null);
  const rows = fenBoard.split("/");
  if (rows.length !== 10) throw new Error("FEN board must have 10 rows, got " + rows.length);
  for (let fenRow = 0; fenRow < 10; fenRow++) {
    // fenRow 0 对应棋盘 row 9（黑底线），fenRow 9 对应 row 0（红底线）
    const boardRow = 9 - fenRow;
    let col = 0;
    for (const ch of rows[fenRow]) {
      if (ch >= "1" && ch <= "9") {
        col += parseInt(ch, 10);
      } else {
        const norm = CHAR_ALIAS[ch] || ch;
        const upper = norm.toUpperCase();
        if (!PIECE_CHARS_RED.has(upper)) throw new Error("Unknown FEN char: " + ch);
        const isRed = norm === upper;
        board[boardRow * 9 + col] = { side: isRed ? "red" : "black", type: upper };
        col++;
      }
    }
    if (col !== 9) throw new Error(`FEN row ${fenRow} has ${col} cols, expected 9`);
  }
  return board;
}

/**
 * 完整 FEN 解析，返回 { board, sideToMove, halfmove, fullmove }
 */
function parseFen(fen) {
  const parts = fen.trim().split(/\s+/);
  const board = parseFenBoard(parts[0]);
  const sideRaw = (parts[1] || "w").toLowerCase();
  const sideToMove = (sideRaw === "w" || sideRaw === "r") ? "red" : "black";
  const halfmove = parseInt(parts[4] || "0", 10);
  const fullmove = parseInt(parts[5] || "1", 10);
  return { board, sideToMove, halfmove, fullmove };
}

// ── 中文棋子字符 ─────────────────────────────────────────────────────────────
const PIECE_CN_RED   = { K:"帅", A:"仕", B:"相", N:"马", R:"车", C:"炮", P:"兵" };
const PIECE_CN_BLACK = { K:"将", A:"士", B:"象", N:"马", R:"车", C:"炮", P:"卒" };

function pieceCN(type, side) {
  return side === "red" ? PIECE_CN_RED[type] : PIECE_CN_BLACK[type];
}

// ── SVG 工具 ─────────────────────────────────────────────────────────────────
const SVG_NS = "http://www.w3.org/2000/svg";

function svgEl(tag, attrs = {}) {
  const el = document.createElementNS(SVG_NS, tag);
  for (const [k, v] of Object.entries(attrs)) el.setAttribute(k, v);
  return el;
}

// ── 棋盘组件 ─────────────────────────────────────────────────────────────────
class XiangqiBoard {
  /**
   * @param {SVGElement} svgEl - 挂载点 SVG 元素
   * @param {object} opts
   *   cellSize: 格距（像素），默认 52
   *   onMove(fromSq, toSq): 用户完成走子回调
   */
  constructor(svg, opts = {}) {
    this.svg = svg;
    this.cellSize = opts.cellSize || 52;
    this.onMove = opts.onMove || null;

    // 布局参数（内部坐标系，默认 cellSize=52 → W/H = 476/524，与 CSS aspect-ratio 兜底一致）
    this.margin = { left: 36, top: 28, right: 24, bottom: 28 };
    this.W = this.margin.left + 8 * this.cellSize + this.margin.right;
    this.H = this.margin.top  + 9 * this.cellSize + this.margin.bottom;

    // 不设固定 width/height 像素属性：显示尺寸由 CSS 约束（#board-container），
    // SVG 按 viewBox 等比缩放；点击换算在 _onClick 里按 getBoundingClientRect 缩放。
    this.svg.removeAttribute("width");
    this.svg.removeAttribute("height");
    this.svg.setAttribute("viewBox", `0 0 ${this.W} ${this.H}`);

    // 状态
    this.board = new Array(90).fill(null);   // sq -> piece|null
    this.flipped = false;                     // 是否翻转（黑方视角）
    this.selectedSq = null;                  // 当前选中格
    this.legalMoves = [];                    // 来自 GameState.legal_moves 的 UCCI 字符串列表
    this.lastMove = null;                    // { from, to } sq
    this.inCheckSide = null;                 // "red"|"black"|null — 将军一方
    this.hintArrow = null;                   // { from, to } sq
    this.hintTopMoves = [];                  // [{move,prob,visits}]
    this.interactive = true;                 // 是否允许交互走子

    // SVG 图层（按绘制顺序）
    this._layers = {};
    for (const id of ["bg","lines","palace","river","coords","highlights",
                       "legal-dots","hint-arrows","pieces","overlay"]) {
      const g = svgEl("g", { class: "layer-" + id });
      svg.appendChild(g);
      this._layers[id] = g;
    }

    // 事件
    this.svg.addEventListener("click", this._onClick.bind(this));
    this.svg.addEventListener("contextmenu", e => { e.preventDefault(); this._clearSelection(); });

    this._drawStatic();
  }

  // ── 坐标变换 ────────────────────────────────────────────────────────────────
  /** row,col -> 视觉 pixel(cx,cy) */
  _rc2px(row, col) {
    const { cellSize: C, margin: M, flipped } = this;
    const vr = flipped ? row       : 9 - row;   // 视觉行（0 = 顶部）
    const vc = flipped ? 8 - col   : col;        // 视觉列（0 = 左侧）
    return { x: M.left + vc * C, y: M.top + vr * C };
  }

  _sq2px(sq) { return this._rc2px(Math.floor(sq / 9), sq % 9); }

  /** pixel -> sq（最近交叉点），若超出返回 -1 */
  _px2sq(px, py) {
    const { cellSize: C, margin: M, flipped } = this;
    const vc = Math.round((px - M.left) / C);
    const vr = Math.round((py - M.top)  / C);
    if (vc < 0 || vc > 8 || vr < 0 || vr > 9) return -1;
    const row = flipped ? vr       : 9 - vr;
    const col = flipped ? 8 - vc   : vc;
    return row * 9 + col;
  }

  // ── 静态棋盘绘制 ────────────────────────────────────────────────────────────
  _drawStatic() {
    this._drawBg();
    this._drawLines();
    this._drawPalace();
    this._drawRiver();
    this._drawCoords();
  }

  _drawBg() {
    const bg = this._layers.bg;
    bg.innerHTML = "";
    // 木纹底色
    bg.appendChild(svgEl("rect", {
      x: 0, y: 0, width: this.W, height: this.H,
      fill: "#d4a96a"
    }));
    // 棋盘区域（略深）
    const { margin: M, cellSize: C } = this;
    bg.appendChild(svgEl("rect", {
      x: M.left - C/2, y: M.top - C/2,
      width: 8*C + C, height: 9*C + C,
      fill: "#c8963e", rx: 4
    }));
  }

  _drawLines() {
    const g = this._layers.lines;
    g.innerHTML = "";
    const { cellSize: C, margin: M } = this;
    const lineStyle = { stroke: "#5a3a00", "stroke-width": "1.2" };

    // 横线 row 0..9（视觉坐标固定，与 flip 无关）
    for (let vr = 0; vr <= 9; vr++) {
      const y = M.top + vr * C;
      const x1 = M.left;
      const x2 = M.left + 8 * C;
      g.appendChild(svgEl("line", { x1, y1: y, x2, y2: y, ...lineStyle }));
    }

    // 竖线：全行仅在边线；中间 7 条在上下两半各自不跨河
    for (let vc = 0; vc <= 8; vc++) {
      const x = M.left + vc * C;
      if (vc === 0 || vc === 8) {
        // 边线贯通
        g.appendChild(svgEl("line", { x1:x, y1:M.top, x2:x, y2:M.top+9*C, ...lineStyle }));
      } else {
        // 上半（黑方，视觉 row 0..4）
        g.appendChild(svgEl("line", { x1:x, y1:M.top,     x2:x, y2:M.top+4*C, ...lineStyle }));
        // 下半（红方，视觉 row 5..9）
        g.appendChild(svgEl("line", { x1:x, y1:M.top+5*C, x2:x, y2:M.top+9*C, ...lineStyle }));
      }
    }
  }

  _drawPalace() {
    const g = this._layers.palace;
    g.innerHTML = "";
    const { cellSize: C, margin: M } = this;
    const lineStyle = { stroke: "#5a3a00", "stroke-width": "1.2" };

    // 在棋盘坐标中，九宫位置固定
    // 红方九宫：row 0-2, col 3-5 -> 视觉（未翻转）row 7-9, col 3-5
    // 黑方九宫：row 7-9, col 3-5 -> 视觉（未翻转）row 0-2, col 3-5
    // 我们直接用视觉坐标绘制（静态，稍后 flip 时重绘）
    const palaces = [
      { vr1: 0, vr2: 2, vc1: 3, vc2: 5 },  // 视觉上方（黑方九宫，未翻转）
      { vr1: 7, vr2: 9, vc1: 3, vc2: 5 },  // 视觉下方（红方九宫，未翻转）
    ];
    for (const { vr1, vr2, vc1, vc2 } of palaces) {
      const x1 = M.left + vc1 * C, y1 = M.top + vr1 * C;
      const x2 = M.left + vc2 * C, y2 = M.top + vr2 * C;
      g.appendChild(svgEl("line", { x1, y1, x2: x2, y2: y2, ...lineStyle }));
      g.appendChild(svgEl("line", { x1: x2, y1, x2: x1, y2: y2, ...lineStyle }));
    }
  }

  _drawRiver() {
    const g = this._layers.river;
    g.innerHTML = "";
    const { cellSize: C, margin: M } = this;
    // 楚河汉界在视觉 row 4 和 row 5 之间
    const y1 = M.top + 4 * C, y2 = M.top + 5 * C;
    const xL = M.left, xR = M.left + 8 * C;
    const yMid = (y1 + y2) / 2;

    // 河界背景（略深色条带）
    g.appendChild(svgEl("rect", {
      x: xL, y: y1, width: xR - xL, height: y2 - y1,
      fill: "#b87333", opacity: "0.35"
    }));

    // 楚河汉界文字（根据翻转调整）
    const textStyle = {
      fill: "#5a3a00", "font-size": `${C * 0.45}px`,
      "font-family": "serif", "text-anchor": "middle",
      "dominant-baseline": "middle", "font-weight": "bold"
    };

    // 未翻转：左=楚河，右=汉界（传统布局）
    // 翻转时互换
    const leftText  = this.flipped ? "汉  界" : "楚  河";
    const rightText = this.flipped ? "楚  河" : "汉  界";

    const t1 = svgEl("text", { x: xL + (xR-xL)*0.25, y: yMid, ...textStyle });
    t1.textContent = leftText;
    const t2 = svgEl("text", { x: xL + (xR-xL)*0.75, y: yMid, ...textStyle });
    t2.textContent = rightText;
    g.appendChild(t1);
    g.appendChild(t2);
  }

  _drawCoords() {
    const g = this._layers.coords;
    g.innerHTML = "";
    const { cellSize: C, margin: M } = this;
    const textStyle = {
      fill: "#5a3a00", "font-size": "11px",
      "font-family": "monospace", "text-anchor": "middle",
      "dominant-baseline": "middle"
    };

    // 列标注（a-i）：底部
    for (let vc = 0; vc <= 8; vc++) {
      const col = this.flipped ? 8 - vc : vc;
      const ch = "abcdefghi"[col];
      const x = M.left + vc * C;
      const t = svgEl("text", { x, y: M.top + 9*C + 14, ...textStyle });
      t.textContent = ch;
      g.appendChild(t);
    }

    // 行标注（0-9）：左侧
    for (let vr = 0; vr <= 9; vr++) {
      const row = this.flipped ? vr : 9 - vr;
      const y = M.top + vr * C;
      const t = svgEl("text", { x: M.left - 20, y, ...textStyle });
      t.textContent = String(row);
      g.appendChild(t);
    }
  }

  // ── 动态内容渲染 ────────────────────────────────────────────────────────────

  /** 刷新高亮（选中 + 最后一着）。落点环画在 overlay 层（棋子之上），
   *  否则会被落点上的棋子盖住只露一圈细边（曾因此看不清落点）。 */
  _renderHighlights() {
    const g = this._layers.highlights;
    const ov = this._layers.overlay;
    g.innerHTML = "";
    ov.innerHTML = "";
    const { cellSize: C } = this;
    const r = C * 0.48;

    const add = (sq, color, opacity = 0.35) => {
      const { x, y } = this._sq2px(sq);
      g.appendChild(svgEl("circle", { cx: x, cy: y, r, fill: color, opacity: String(opacity) }));
    };

    if (this.lastMove) {
      add(this.lastMove.from, "#f1c40f", 0.5);
      add(this.lastMove.to,   "#f1c40f", 0.5);

      // 落点环：内环贴棋子外缘描边 + 外环淡光晕
      const { x, y } = this._sq2px(this.lastMove.to);
      const R = C * 0.42;  // 与 _renderPieces 的棋子半径一致
      const ring = svgEl("g", { class: "last-move-ring" });
      ring.appendChild(svgEl("circle", {
        cx: x, cy: y, r: R + 3,
        fill: "none", stroke: "#f39c12", "stroke-width": "3", opacity: "0.95"
      }));
      ring.appendChild(svgEl("circle", {
        cx: x, cy: y, r: R + 6.5,
        fill: "none", stroke: "#f39c12", "stroke-width": "2.5", opacity: "0.3"
      }));
      ov.appendChild(ring);
    }
    if (this.selectedSq !== null) {
      add(this.selectedSq, "#3498db", 0.6);
    }
  }

  /** 渲染合法落点小圆点 */
  _renderLegalDots() {
    const g = this._layers["legal-dots"];
    g.innerHTML = "";
    if (this.selectedSq === null) return;

    const { cellSize: C } = this;
    const fromStr = this._sq2ucci(this.selectedSq);
    const dests = new Set();

    for (const mv of this.legalMoves) {
      if (mv.slice(0, 2) === fromStr) {
        dests.add(this._ucci2sq(mv.slice(2, 4)));
      }
    }

    for (const sq of dests) {
      const { x, y } = this._sq2px(sq);
      const hasPiece = this.board[sq] !== null;
      if (hasPiece) {
        // 吃子：空心圆
        g.appendChild(svgEl("circle", {
          cx: x, cy: y, r: C * 0.46,
          fill: "none", stroke: "#3498db", "stroke-width": "3", opacity: "0.75"
        }));
      } else {
        // 空格：实心小圆
        g.appendChild(svgEl("circle", {
          cx: x, cy: y, r: C * 0.14,
          fill: "#3498db", opacity: "0.8"
        }));
      }
    }
  }

  /** 渲染棋子 */
  _renderPieces() {
    const g = this._layers.pieces;
    g.innerHTML = "";
    const { cellSize: C } = this;
    const R = C * 0.42;

    for (let sq = 0; sq < 90; sq++) {
      const piece = this.board[sq];
      if (!piece) continue;

      const { x, y } = this._sq2px(sq);
      const isRed = piece.side === "red";
      const pieceColor   = isRed ? "#c0392b" : "#2c3e50";
      const bgColor      = "#f5e6c8";  // 米色木纹底
      const borderColor  = isRed ? "#8b1a1a" : "#1a252f";
      const isInCheck    = this.inCheckSide === piece.side && piece.type === "K";

      const grp = svgEl("g", { class: "piece" + (isInCheck ? " piece-in-check" : "") });

      // 外圈阴影
      grp.appendChild(svgEl("circle", {
        cx: x, cy: y+1, r: R, fill: "rgba(0,0,0,0.25)"
      }));
      // 底色
      grp.appendChild(svgEl("circle", { cx: x, cy: y, r: R, fill: bgColor }));
      // 边框
      grp.appendChild(svgEl("circle", {
        cx: x, cy: y, r: R,
        fill: "none", stroke: borderColor, "stroke-width": "2.5"
      }));
      // 内圈装饰
      grp.appendChild(svgEl("circle", {
        cx: x, cy: y, r: R * 0.82,
        fill: "none", stroke: pieceColor, "stroke-width": "1.2"
      }));

      // 中文字
      const txt = svgEl("text", {
        x, y: y + 1,
        fill: pieceColor,
        "font-size": `${R * 1.1}px`,
        "font-family": '"Microsoft YaHei","PingFang SC","Noto Sans CJK SC",serif',
        "font-weight": "bold",
        "text-anchor": "middle",
        "dominant-baseline": "middle",
        "pointer-events": "none"
      });
      txt.textContent = pieceCN(piece.type, piece.side);
      grp.appendChild(txt);

      g.appendChild(grp);
    }
  }

  /** 渲染提示箭头 */
  _renderHintArrows() {
    const g = this._layers["hint-arrows"];
    g.innerHTML = "";
    if (!this.hintArrow) return;

    const { x: x1, y: y1 } = this._sq2px(this.hintArrow.from);
    const { x: x2, y: y2 } = this._sq2px(this.hintArrow.to);
    const { cellSize: C } = this;

    // 箭头 marker
    const markerId = "hint-arrow-marker";
    let defs = this.svg.querySelector("defs");
    if (!defs) {
      defs = svgEl("defs");
      this.svg.insertBefore(defs, this.svg.firstChild);
    }
    // 每次重建 marker（避免 id 冲突问题）
    const existingMarker = defs.querySelector("#" + markerId);
    if (existingMarker) existingMarker.remove();

    const marker = svgEl("marker", {
      id: markerId, markerWidth: "6", markerHeight: "6",
      refX: "5", refY: "3", orient: "auto"
    });
    marker.appendChild(svgEl("path", { d: "M0,0 L6,3 L0,6 Z", fill: "#27ae60", opacity: "0.9" }));
    defs.appendChild(marker);

    // 计算箭头（缩进避免遮挡棋子）
    const dx = x2 - x1, dy = y2 - y1;
    const len = Math.sqrt(dx*dx + dy*dy);
    const ux = dx/len, uy = dy/len;
    const shrink = C * 0.38;
    const ax1 = x1 + ux*shrink, ay1 = y1 + uy*shrink;
    const ax2 = x2 - ux*shrink, ay2 = y2 - uy*shrink;

    g.appendChild(svgEl("line", {
      x1: ax1, y1: ay1, x2: ax2, y2: ay2,
      stroke: "#27ae60", "stroke-width": "4", opacity: "0.85",
      "marker-end": `url(#${markerId})`
    }));
  }

  // ── 公开 API ────────────────────────────────────────────────────────────────

  /** 从 FEN 重绘棋盘（不改变交互状态） */
  setFen(fen) {
    try {
      const parsed = parseFen(fen);
      this.board = parsed.board;
    } catch (e) {
      console.error("FEN 解析失败:", e);
    }
    this._clearSelection();
    this._render();
  }

  /** 设置合法着法（来自 GameState.legal_moves，UCCI 字符串数组） */
  setLegalMoves(moves) {
    this.legalMoves = moves || [];
  }

  /** 设置最后一着高亮 */
  setLastMove(ucciOrNull) {
    if (!ucciOrNull) {
      this.lastMove = null;
    } else {
      this.lastMove = {
        from: this._ucci2sq(ucciOrNull.slice(0, 2)),
        to:   this._ucci2sq(ucciOrNull.slice(2, 4))
      };
    }
    this._renderHighlights();
  }

  /** 设置将军一方（用于闪烁） */
  setInCheck(side) {
    this.inCheckSide = side || null;
    this._renderPieces();
  }

  /** 设置提示箭头和 top moves */
  setHint(ucciOrNull, topMoves) {
    this.hintArrow = ucciOrNull ? {
      from: this._ucci2sq(ucciOrNull.slice(0,2)),
      to:   this._ucci2sq(ucciOrNull.slice(2,4))
    } : null;
    this.hintTopMoves = topMoves || [];
    this._renderHintArrows();
  }

  /** 翻转棋盘视角 */
  flip() {
    this.flipped = !this.flipped;
    this._clearSelection();
    // 重绘所有静态层
    this._drawLines();
    this._drawPalace();
    this._drawRiver();
    this._drawCoords();
    this._render();
  }

  /** 是否允许走子交互 */
  setInteractive(enabled) {
    this.interactive = enabled;
    if (!enabled) this._clearSelection();
  }

  /** 完整重绘（FEN 变化后调用） */
  _render() {
    this._renderHighlights();
    this._renderLegalDots();
    this._renderPieces();
    this._renderHintArrows();
  }

  // ── 点击交互 ────────────────────────────────────────────────────────────────
  _onClick(e) {
    if (!this.interactive) return;

    const rect = this.svg.getBoundingClientRect();
    const scaleX = this.W / rect.width;
    const scaleY = this.H / rect.height;
    const px = (e.clientX - rect.left) * scaleX;
    const py = (e.clientY - rect.top)  * scaleY;
    const sq = this._px2sq(px, py);
    if (sq === -1) return;

    if (this.selectedSq === null) {
      // 选中有棋子的格
      if (this.board[sq]) {
        this.selectedSq = sq;
        this._renderHighlights();
        this._renderLegalDots();
      }
    } else {
      if (sq === this.selectedSq) {
        // 取消选中
        this._clearSelection();
        return;
      }

      // 检查是否为合法着法
      const fromStr = this._sq2ucci(this.selectedSq);
      const toStr   = this._sq2ucci(sq);
      const move    = fromStr + toStr;
      if (this.legalMoves.includes(move)) {
        const from = this.selectedSq;
        this._clearSelection();
        if (this.onMove) this.onMove(from, sq);
      } else if (this.board[sq] && this._isOwnPiece(sq)) {
        // 切换选中到己方另一棋子
        this.selectedSq = sq;
        this._renderHighlights();
        this._renderLegalDots();
      } else {
        this._clearSelection();
      }
    }
  }

  _clearSelection() {
    this.selectedSq = null;
    this._renderHighlights();
    this._renderLegalDots();
  }

  /** 判断 sq 上的棋子是否与当前 selectedSq 同色 */
  _isOwnPiece(sq) {
    if (this.selectedSq === null || !this.board[sq]) return false;
    const sel = this.board[this.selectedSq];
    return sel && this.board[sq].side === sel.side;
  }

  // ── UCCI 坐标工具 ────────────────────────────────────────────────────────────
  _ucci2sq(s) {
    const col = "abcdefghi".indexOf(s[0].toLowerCase());
    const row = parseInt(s[1], 10);
    return row * 9 + col;
  }

  _sq2ucci(sq) {
    const row = Math.floor(sq / 9), col = sq % 9;
    return "abcdefghi"[col] + String(row);
  }
}

// ── 导出（全局） ─────────────────────────────────────────────────────────────
window.XiangqiBoard = XiangqiBoard;
window.parseFen = parseFen;
window.parseFenBoard = parseFenBoard;
