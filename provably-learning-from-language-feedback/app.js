/* =====================================================================
   HELiX blog — interactive demos
   - Battleship (language feedback, hidden reward)
   - Minesweeper (language feedback, hidden reward)
   - Score-matrix sandbox (the HELiX decision rule, made visible)
   ===================================================================== */

const ROWS = "ABCDEF";
const N = 6;
const cellName = (r, c) => ROWS[r] + (c + 1);

function el(tag, cls, txt) {
  const e = document.createElement(tag);
  if (cls) e.className = cls;
  if (txt !== undefined) e.textContent = txt;
  return e;
}
function clamp01(x) { return Math.max(0, Math.min(1, x)); }
function choice(arr) { return arr[Math.floor(Math.random() * arr.length)]; }
function shuffle(a) { for (let i = a.length - 1; i > 0; i--) { const j = (Math.random() * (i + 1)) | 0; [a[i], a[j]] = [a[j], a[i]]; } return a; }
function cap(s) { return s ? s.charAt(0).toUpperCase() + s.slice(1) : s; }

/* ======================================================================
   BATTLESHIP
   ====================================================================== */
const Battleship = (() => {
  const SHIPS = [5, 4, 3];
  let grid, ships, shots, sunkCount, reward, over, turns;

  function placeShips() {
    grid = Array.from({ length: N }, () => Array(N).fill(0)); // 0 empty
    ships = [];
    for (const len of SHIPS) {
      let placed = false, guard = 0;
      while (!placed && guard++ < 500) {
        const horiz = Math.random() < 0.5;
        const r = (Math.random() * N) | 0;
        const c = (Math.random() * N) | 0;
        const cells = [];
        for (let k = 0; k < len; k++) {
          const rr = horiz ? r : r + k;
          const cc = horiz ? c + k : c;
          if (rr >= N || cc >= N || grid[rr][cc] !== 0) { cells.length = 0; break; }
          cells.push([rr, cc]);
        }
        if (cells.length === len) {
          const id = ships.length + 1;
          cells.forEach(([rr, cc]) => grid[rr][cc] = id);
          ships.push({ id, len, cells, hits: 0 });
          placed = true;
        }
      }
    }
  }

  function reset() {
    placeShips();
    shots = Array.from({ length: N }, () => Array(N).fill(0)); // 0 none,1 miss,2 hit
    sunkCount = 0; reward = 0; over = false; turns = 0;
    render();
    log(`<span class="muted">New game — three ships hidden (lengths 5, 4, 3). Fire a cell to begin. The reward is hidden from the agent.</span>`, true);
    status("find the ships", "explore");
    stat(); sync();
  }

  function fire(r, c) {
    if (over || shots[r][c] !== 0) return;
    turns++;
    const id = grid[r][c];
    if (id === 0) {
      shots[r][c] = 1;
      log(`<b>${cellName(r, c)}</b> → <span class="fb">Miss.</span> Nothing there.`);
    } else {
      shots[r][c] = 2;
      const ship = ships.find(s => s.id === id);
      ship.hits++;
      if (ship.hits === ship.len) {
        sunkCount++;
        reward += 1.0;
        log(`<b>${cellName(r, c)}</b> → <span class="fb">Hit and SUNK!</span> A ship is destroyed. <span class="rw">[hidden reward +1.0]</span>`);
      } else {
        reward += 0.5;
        log(`<b>${cellName(r, c)}</b> → <span class="fb">Hit!</span> A ship was struck but not sunk. <span class="rw">[hidden reward +0.5]</span>`);
      }
    }
    if (sunkCount === SHIPS.length) {
      over = true;
      log(`<span class="fb">All ships sunk in ${turns} shots!</span> Final hidden reward <span class="rw">${reward.toFixed(1)}</span>.`);
      status("victory 🎉", "exploit");
    } else if (turns >= 20) {
      over = true;
      revealAll();
      log(`<span class="fb">Out of turns.</span> ${sunkCount}/3 ships sunk. Hidden reward <span class="rw">${reward.toFixed(1)}</span>.`);
      status("out of turns", "explore");
    }
    render(); stat(); sync();
  }

  function revealAll() {
    for (let r = 0; r < N; r++) for (let c = 0; c < N; c++)
      if (grid[r][c] !== 0 && shots[r][c] === 0) shots[r][c] = 3; // unrevealed ship
  }

  function render() {
    const host = document.getElementById("bs-board");
    host.innerHTML = "";
    const board = el("div", "board");
    board.style.gridTemplateColumns = `repeat(${N + 1}, auto)`;
    board.appendChild(el("div", "cell label", ""));
    for (let c = 0; c < N; c++) board.appendChild(el("div", "cell label", String(c + 1)));
    for (let r = 0; r < N; r++) {
      board.appendChild(el("div", "cell label", ROWS[r]));
      for (let c = 0; c < N; c++) {
        const s = shots[r][c];
        const b = el("button", "cell");
        if (s === 1) { b.classList.add("miss"); b.textContent = "·"; b.disabled = true; }
        else if (s === 2) { b.classList.add(ships.find(sh => sh.id === grid[r][c]).hits === ships.find(sh => sh.id === grid[r][c]).len ? "sunk" : "hit"); b.textContent = "✕"; b.disabled = true; }
        else if (s === 3) { b.classList.add("miss"); b.textContent = "○"; b.disabled = true; b.style.opacity = .5; }
        else { b.textContent = ""; b.onclick = () => fire(r, c); if (over) b.disabled = true; }
        board.appendChild(b);
      }
    }
    host.appendChild(board);
  }

  function log(html, clear) {
    const host = document.getElementById("bs-log");
    if (clear) host.innerHTML = "";
    const line = el("div", "fl"); line.innerHTML = html;
    host.prepend(line);
  }
  function status(txt, cls) {
    const s = document.getElementById("bs-status");
    s.textContent = txt; s.className = "pill " + cls;
  }
  function stat() {
    document.getElementById("bs-stat").innerHTML =
      `Shots: <b>${turns}/20</b> · Ships sunk: <b>${sunkCount}/3</b> · <span class="muted">hidden reward</span> <b>${reward.toFixed(1)}</b>`;
  }

  // Push this board to the score-matrix sandbox so the two stay in sync.
  function sync() { ScoreMatrix.syncFromBattleship(boardState()); }

  /* "What would HELiX do?" — the sandbox already mirrors this board; just scroll to it */
  function suggest() {
    sync();
    document.getElementById("matrix").scrollIntoView({ behavior: "smooth", block: "start" });
  }
  function boardState() {
    // ship context for the sandbox's hypotheses
    const sunk = ships.filter(s => s.hits === s.len).map(s => s.len).sort((a, b) => a - b);
    const remaining = ships.filter(s => s.hits < s.len).map(s => s.len).sort((a, b) => b - a);
    // active hits = struck cells on ships not yet sunk (exclude sunk-ship cells)
    const activeHits = [];
    for (let r = 0; r < N; r++) for (let c = 0; c < N; c++) {
      if (shots[r][c] !== 2) continue;
      const sh = ships.find(s => s.id === grid[r][c]);
      if (sh && sh.hits < sh.len) activeHits.push([r, c]);
    }
    // cells belonging to fully-sunk ships (so the sandbox can colour them like the game)
    const sunkCells = [];
    ships.filter(s => s.hits === s.len).forEach(s => s.cells.forEach(c => sunkCells.push(c)));
    return { shots: shots.map(r => r.slice()), turns, sunk, remaining, activeHits, over, sunkCells };
  }

  return { reset, suggest, fire };
})();

/* ======================================================================
   MINESWEEPER
   ====================================================================== */
const Minesweeper = (() => {
  const MINES = 6;
  let mine, adj, revealed, flagged, over, won, reward, safeRevealed, flagMode;

  function neighbors(r, c) {
    const out = [];
    for (let dr = -1; dr <= 1; dr++) for (let dc = -1; dc <= 1; dc++) {
      if (dr === 0 && dc === 0) continue;
      const rr = r + dr, cc = c + dc;
      if (rr >= 0 && rr < N && cc >= 0 && cc < N) out.push([rr, cc]);
    }
    return out;
  }

  function reset() {
    mine = Array.from({ length: N }, () => Array(N).fill(false));
    let placed = 0;
    while (placed < MINES) {
      const r = (Math.random() * N) | 0, c = (Math.random() * N) | 0;
      if (!mine[r][c]) { mine[r][c] = true; placed++; }
    }
    adj = Array.from({ length: N }, () => Array(N).fill(0));
    for (let r = 0; r < N; r++) for (let c = 0; c < N; c++)
      if (!mine[r][c]) adj[r][c] = neighbors(r, c).filter(([rr, cc]) => mine[rr][cc]).length;
    revealed = Array.from({ length: N }, () => Array(N).fill(false));
    flagged = Array.from({ length: N }, () => Array(N).fill(false));
    over = false; won = false; reward = 0; safeRevealed = 0; flagMode = false;
    updateFlagBtn();
    render();
    log(`<span class="muted">New game — ${MINES} mines hidden on the 6×6 grid. Reveal a cell to begin.</span>`, true);
    status("clear the board", "explore");
    stat();
  }

  function cascade(r, c) {
    const stack = [[r, c]];
    while (stack.length) {
      const [cr, cc] = stack.pop();
      if (revealed[cr][cc] || flagged[cr][cc]) continue;
      revealed[cr][cc] = true; safeRevealed++; reward += 0.2;
      if (adj[cr][cc] === 0) neighbors(cr, cc).forEach(([rr, ccc]) => { if (!revealed[rr][ccc]) stack.push([rr, ccc]); });
    }
  }

  function reveal(r, c) {
    if (over) return;
    if (flagMode) { toggleFlag(r, c); return; }
    if (revealed[r][c] || flagged[r][c]) {
      reward -= 0.2;
      log(`<b>${cellName(r, c)}</b> → <span class="fb">Invalid move.</span> Cell already settled. <span class="rw">[hidden reward −0.2]</span>`);
      render(); stat(); return;
    }
    if (mine[r][c]) {
      revealed[r][c] = true; over = true;
      log(`<b>${cellName(r, c)}</b> → <span class="fb">BOOM — a mine.</span> Game over. Hidden reward <span class="rw">${reward.toFixed(1)}</span>.`);
      status("hit a mine 💥", "explore");
      revealMines(); render(); stat(); return;
    }
    const before = safeRevealed;
    cascade(r, c);
    const gained = safeRevealed - before;
    const n = adj[r][c];
    const clue = n === 0 ? `It's blank — ${gained} cells opened up automatically.` : `${n} mine${n > 1 ? "s" : ""} touch this cell.`;
    log(`<b>${cellName(r, c)}</b> → <span class="fb">Safe.</span> ${clue} <span class="rw">[hidden reward +${(gained * 0.2).toFixed(1)}]</span>`);
    if (safeRevealed === N * N - MINES) {
      over = true; won = true; reward += 1.0;
      log(`<span class="fb">Board cleared!</span> Every safe cell revealed. Final hidden reward <span class="rw">${reward.toFixed(1)}</span>.`);
      status("solved 🎉", "exploit");
    }
    render(); stat();
  }

  function toggleFlag(r, c) {
    if (over || revealed[r][c]) return;
    flagged[r][c] = !flagged[r][c];
    render();
  }
  function revealMines() {
    for (let r = 0; r < N; r++) for (let c = 0; c < N; c++) if (mine[r][c]) revealed[r][c] = "mine";
  }

  function render() {
    const host = document.getElementById("ms-board");
    host.innerHTML = "";
    const board = el("div", "board");
    board.style.gridTemplateColumns = `repeat(${N + 1}, auto)`;
    board.appendChild(el("div", "cell label", ""));
    for (let c = 0; c < N; c++) board.appendChild(el("div", "cell label", String(c + 1)));
    for (let r = 0; r < N; r++) {
      board.appendChild(el("div", "cell label", ROWS[r]));
      for (let c = 0; c < N; c++) {
        const b = el("button", "cell");
        if (revealed[r][c] === "mine") { b.classList.add("mine"); b.textContent = "✸"; b.disabled = true; }
        else if (revealed[r][c]) { b.classList.add("safe", "n" + adj[r][c]); b.textContent = adj[r][c] || ""; b.disabled = true; }
        else if (flagged[r][c]) { b.classList.add("flag"); b.textContent = "⚑"; b.onclick = () => reveal(r, c); b.oncontextmenu = (e) => { e.preventDefault(); toggleFlag(r, c); }; }
        else {
          b.textContent = "";
          b.onclick = () => reveal(r, c);
          b.oncontextmenu = (e) => { e.preventDefault(); toggleFlag(r, c); };
          if (over) b.disabled = true;
        }
        board.appendChild(b);
      }
    }
    host.appendChild(board);
  }

  function log(html, clear) {
    const host = document.getElementById("ms-log");
    if (clear) host.innerHTML = "";
    const line = el("div", "fl"); line.innerHTML = html;
    host.prepend(line);
  }
  function status(txt, cls) { const s = document.getElementById("ms-status"); s.textContent = txt; s.className = "pill " + cls; }
  function stat() {
    document.getElementById("ms-stat").innerHTML =
      `Safe revealed: <b>${safeRevealed}/${N * N - MINES}</b> · <span class="muted">hidden reward</span> <b>${reward.toFixed(1)}</b>`;
  }
  function updateFlagBtn() {
    document.getElementById("ms-flag").textContent = `🚩 Flag mode: ${flagMode ? "on" : "off"}`;
  }
  function toggleFlagMode() { flagMode = !flagMode; updateFlagBtn(); }

  return { reset, toggleFlagMode };
})();

/* ======================================================================
   SCORE-MATRIX SANDBOX  (the HELiX decision rule, made visible)

   The "LLM" is a transparent heuristic:
   - Each hypothesis claims a TARGET cell + a region/line on the board.
   - R(hypothesis, action) scores how well the action aligns with that
     hypothesis's intent (1.0 for the hypothesis's own target, decaying
     with distance / off-line). The diagonal is ~1.0 by construction,
     mirroring the paper.
   - Stage machine: sample -> fill matrix -> row argmaxes ->
     consensus? exploit : (eliminate + tie-break w/ ref policy) explore.
   ====================================================================== */
const ScoreMatrix = (() => {
  let board;          // {shots[][], turns}  (1 miss, 2 hit, 0 none)
  let hyps = [];      // [{text, target:[r,c], line:[[r,c]...], color, permissive?}]
  let actions = [];   // [{name, rc:[r,c], ref:bool}]
  let S = [];         // raw score matrix [h][a]
  let stage = 0;      // 0 none, 1 sampled, 2 filled, 3 argmax, 4(=rescore|decide), 5 decide
  let decision = null;
  let usePiref = false;   // optional π_ref re-scoring extension (off by default = core HELiX)
  let consensusFound = false;
  let needsRescore = false;   // usePiref && no consensus
  let baseline = [];      // per-hypothesis baseline b_η = mean of its random-action scores
  let A = [];             // advantage matrix  A[h][a] = S[h][a] - baseline[h]
  let rescored = false;   // whether the matrix is currently shown as advantages

  const SHIP_LENS = [5, 4, 3];
  // A short phrase about which ships are sunk / remain, so the sampled hypotheses
  // can reason about ship lengths and how far a streak must extend to fit.
  function shipCtx() {
    const sunk = board.sunk || [];
    const rem = board.remaining || SHIP_LENS;
    const minRem = board.minRem != null ? board.minRem : Math.min(...rem);
    const sl = board.streakLen || 2;
    const phrase = sunk.length === 0
      ? `no ships sunk yet (lengths ${rem.join(', ')} still out there)`
      : `the ${sunk.join('- and ')}-cell ship${sunk.length > 1 ? "s are" : " is"} already sunk, leaving the ${rem.join("- and ")}-cell`;
    return { sunk, rem, minRem, sl, phrase, need: Math.max(1, minRem - sl) };
  }

  function unfired() {
    const out = [];
    for (let r = 0; r < N; r++) for (let c = 0; c < N; c++) if (board.shots[r][c] === 0) out.push([r, c]);
    return out;
  }
  function hitCells() {
    // on a live board, only the active (non-sunk) hits form the streak we reason about
    if (board.activeHits) return board.activeHits.map(c => c.slice());
    const out = [];
    for (let r = 0; r < N; r++) for (let c = 0; c < N; c++) if (board.shots[r][c] === 2) out.push([r, c]);
    return out;
  }

  /* ---- sample hypotheses + candidate actions from the board ---- */
  function sampleHypotheses() {
    hyps = []; actions = [];
    const hits = hitCells();
    const open = unfired();
    const seen = new Set();
    const addAction = (rc, ref) => {
      const nm = cellName(rc[0], rc[1]);
      if (seen.has(nm)) return; seen.add(nm);
      actions.push({ name: nm, rc, ref: !!ref });
    };

    const cx = shipCtx();
    const inGrid = (r, c) => r >= 0 && r < N && c >= 0 && c < N;
    const isOpen = (r, c) => inGrid(r, c) && board.shots[r][c] === 0;
    const rnd = v => Math.round(v * 100) / 100;
    const freeOpen = () => shuffle(open.filter(([r, c]) => !seen.has(cellName(r, c))));

    // ---- hidden belief builders: each returns { cellName: prob } over OPEN water ----
    // a decaying ray of belief stepping (dr,dc) across open water from (r,c)
    function ray(r, c, dr, dc, startP, decay) {
      const b = {}; let p = startP;
      while (isOpen(r, c) && p >= 0.12) { b[cellName(r, c)] = rnd(p); p *= decay; r += dr; c += dc; }
      return b;
    }
    // a blob of belief around a centre, decaying with Manhattan distance (broad if decay~1)
    function blob(cr, cc, peak, decay) {
      const b = {};
      for (const [r, c] of open) {
        const p = peak * Math.pow(decay, Math.abs(r - cr) + Math.abs(c - cc));
        if (p >= 0.08) b[cellName(r, c)] = rnd(p);
      }
      return b;
    }
    // a flat belief: the ship could be anywhere still open
    const flat = (val) => { const b = {}; for (const [r, c] of open) b[cellName(r, c)] = val; return b; };
    const maxMerge = (...bs) => { const b = {}; for (const bb of bs) for (const k in bb) b[k] = Math.max(b[k] || 0, bb[k]); return b; };

    // register a hypothesis: `belief` is the hidden distribution; `target` (its argmax)
    // is the candidate-action column the hypothesis "proposes". scoreOf just looks up belief.
    function addHyp(text, belief, target, opts) {
      const nm = cellName(target[0], target[1]);
      if (belief[nm] === undefined) belief[nm] = 1.0;     // guarantee the proposed cell is scored
      hyps.push({ text, belief, target, permissive: !!(opts && opts.permissive) });
      addAction(target);
    }

    // ---- 1. extension hypotheses from the hit streak (sharp, concentrated belief) ----
    let openEnds = [];
    if (hits.length >= 1) {
      const horiz = board.hitLine.horiz;
      const sorted = hits.slice().sort((a, b) => horiz ? a[1] - b[1] : a[0] - b[0]);
      const first = sorted[0], last = sorted[sorted.length - 1];

      if (hits.length >= 2) {
        const cand = horiz
          ? [{ e: [first[0], first[1] - 1], d: [0, -1] }, { e: [last[0], last[1] + 1], d: [0, 1] }]
          : [{ e: [first[0] - 1, first[1]], d: [-1, 0] }, { e: [last[0] + 1, last[1]], d: [1, 0] }];
        openEnds = cand.filter(({ e }) => isOpen(e[0], e[1]));
        openEnds.forEach(({ e, d }) => addHyp(
          `${cap(cx.phrase)}. The ${cx.minRem}-cell ship runs ${horiz ? "horizontally" : "vertically"}; it likely extends to <b>${cellName(e[0], e[1])}</b>.`,
          ray(e[0], e[1], d[0], d[1], 1.0, 0.5), e));
      } else {
        // single hit -> separate "runs horizontally" and "runs vertically" guesses
        [{ dir: "horizontally", rays: [[0, 1], [0, -1]] }, { dir: "vertically", rays: [[1, 0], [-1, 0]] }].forEach(({ dir, rays }) => {
          let belief = {}, target = null;
          rays.forEach(([dr, dc]) => {
            const er = first[0] + dr, ec = first[1] + dc;
            if (isOpen(er, ec)) { belief = maxMerge(belief, ray(er, ec, dr, dc, 1.0, 0.55)); if (!target) target = [er, ec]; }
          });
          if (target) { addHyp(`${cap(cx.phrase)}. This hit could be a ship running <b>${dir}</b> — try <b>${cellName(target[0], target[1])}</b>.`, belief, target); openEnds.push({ e: target }); }
        });
      }

      // forced single extension -> a SECOND confirming hypothesis on the same square -> consensus
      if (hits.length >= 2 && openEnds.length === 1) {
        const e = openEnds[0].e;
        addHyp(`Counting the ${cx.sl} hits, a ${cx.minRem}-cell ship isn't finished — <b>${cellName(e[0], e[1])}</b> is the only square that completes it.`,
          blob(e[0], e[1], 1.0, 0.45), e);
      }
    }

    const forced = hits.length >= 2 && openEnds.length === 1;

    // ---- pad to >=2 hypotheses on a sparse/empty board (region guesses) ----
    while (hyps.length < 2) {
      const t = freeOpen()[0];
      if (!t) break;
      addHyp(`Unexplored area large enough for a ${cx.minRem}-cell ship — one may be hiding near <b>${cellName(t[0], t[1])}</b>.`, blob(t[0], t[1], 1.0, 0.6), t);
    }

    // ---- always reach 3 hypotheses by adding ONE that SPREADS its belief ----
    // On a forced-consensus board it's a FLAT guess (equal odds on every open square): its
    // argmax is *every* action, so it still rates the consensus cell top -> consensus survives.
    // Otherwise it's a broad blob — it adds off-diagonal variation and is the row π_ref
    // re-scoring demotes (its belief leaks onto the random reference cells).
    if (hyps.length < 3 && open.length) {
      if (forced) {
        const t = freeOpen()[0] || open[0];
        addHyp(`Honestly no idea which ship or exactly where — every open square looks about equally likely.`, flat(0.3), t);
        // H1 & H2 both proposed the same forced square, so we'd only have 2 action columns.
        // Sample a 3rd random square (distinct from the others) so the matrix has three actions;
        // its column is scored straight from each hypothesis's hidden belief dict. Prefer a
        // square the sharp hypotheses actually believe in, so the column shows real variation.
        const opinion = new Set();
        hyps.forEach(h => { for (const k in h.belief) if (h.belief[k] > 0.35) opinion.add(k); });
        let nm = shuffle([...opinion].filter(k => !seen.has(k)))[0];
        if (!nm) { const e = freeOpen()[0]; if (e) nm = cellName(e[0], e[1]); }
        if (nm) addAction([ROWS.indexOf(nm[0]), parseInt(nm.slice(1), 10) - 1]);
      } else {
        const far = freeOpen().filter(([r, c]) => !hits.length || hits.every(h => Math.abs(h[0] - r) + Math.abs(h[1] - c) >= 3));
        const t = far[0] || freeOpen()[0];
        if (t) addHyp(`A separate unsunk ship might be sitting around <b>${cellName(t[0], t[1])}</b> — a fuzzy guess over a wide area.`, blob(t[0], t[1], 1.0, 0.82), t);
      }
    }

    addRefActions();
    assignColors();
  }

  // π_ref proposes M=2 random valid actions — only when the extension is enabled.
  function addRefActions() {
    if (!usePiref) return;
    const seen = new Set(actions.map(a => a.name));
    let added = 0;
    const pool = shuffle(unfired().filter(rc => !seen.has(cellName(rc[0], rc[1]))));
    for (const rc of pool) {
      if (added >= 2) break;
      actions.push({ name: cellName(rc[0], rc[1]), rc, ref: true }); added++;
    }
  }

  const HCOLORS = ["#b6543a", "#4e6477", "#5f6f4e", "#8a6d3b", "#6a5a8a"];
  function assignColors() { hyps.forEach((h, i) => { h.color = HCOLORS[i % HCOLORS.length]; }); }

  /* ---- the heuristic "LLM" reward mapping R(eta, a) ----
     Score = how much this hypothesis BELIEVES the ship occupies the action's cell.
     It is just a lookup in the hypothesis's hidden belief dict (0 if not present). */
  function scoreOf(h, a) {
    const v = h.belief[a.name];
    return v === undefined ? 0 : v;
  }

  function buildMatrix() {
    S = hyps.map(h => actions.map(a => scoreOf(h, a)));
    // consensus is read off the RAW scores (a shared row-argmax across all hypotheses)
    const sets = rowArgmaxSets(S);
    let inter = [...sets[0]];
    for (let i = 1; i < sets.length; i++) inter = inter.filter(j => sets[i].has(j));
    consensusFound = inter.length > 0;
    needsRescore = usePiref && !consensusFound;   // re-scoring only changes the explore step
  }

  // Re-score: subtract each hypothesis's baseline (its mean score on the random
  // π_ref actions) from its row, turning raw scores into advantages.
  function doRescore() {
    const r2 = x => Math.round(x * 100) / 100;                 // keep everything at 2 decimals
    const refIdx = actions.map((a, j) => a.ref ? j : -1).filter(j => j >= 0);
    baseline = S.map(row => refIdx.length ? r2(refIdx.reduce((s, j) => s + row[j], 0) / refIdx.length) : 0);
    // Subtract bₕ ONLY from the real action columns. The π_ref reference columns keep their
    // raw belief (they're the values bₕ is averaged from), so the subtraction stays visible.
    A = S.map((row, i) => row.map((v, j) => actions[j].ref ? r2(v) : r2(v - baseline[i])));
    rescored = true;
  }

  /* ---- render board ---- */
  function renderBoard() {
    const host = document.getElementById("sm-board");
    host.innerHTML = "";
    const wrap = el("div", "board");
    wrap.style.gridTemplateColumns = `repeat(${N + 1}, auto)`;
    wrap.appendChild(el("div", "cell label", ""));
    for (let c = 0; c < N; c++) wrap.appendChild(el("div", "cell label", String(c + 1)));
    for (let r = 0; r < N; r++) {
      wrap.appendChild(el("div", "cell label", ROWS[r]));
      for (let c = 0; c < N; c++) {
        const b = el("div", "cell");
        const s = board.shots[r][c];
        if (s === 1) { b.classList.add("miss"); b.textContent = "·"; }
        else if (s === 2) { b.classList.add(board.sunkCells && board.sunkCells.has(r + "," + c) ? "sunk" : "hit"); b.textContent = "✕"; }
        else if (s === 3) { b.classList.add("miss"); b.textContent = "○"; b.style.opacity = ".5"; }  // revealed ship at game over
        else {
          // unfired cell — clickable to fire (keeps both boards in sync), unless game over
          if (!board.over) { b.style.cursor = "pointer"; b.onclick = () => Battleship.fire(r, c); }
          // annotate candidate actions on the board
          const a = actions.find(x => x.rc[0] === r && x.rc[1] === c);
          if (a && stage >= 1 && !board.over) {
            const isPick = decision && actions[decision.actIdx] && a === actions[decision.actIdx];
            // which hypothesis proposed this action? label with its index (1-based)
            const hi = hyps.findIndex(h => h.target && h.target[0] === r && h.target[1] === c);
            if (isPick) {
              // HELiX's chosen shot, shown on the board
              b.style.background = "var(--clay)";
              b.style.color = "#fff";
              b.style.outline = "2px solid var(--clay)";
              b.style.outlineOffset = "-2px";
              b.textContent = "◎";
              b.title = "HELiX fires here";
            } else {
              const col = hi >= 0 ? hyps[hi].color : "var(--ink-faint)";
              b.textContent = hi >= 0 ? String(hi + 1) : "✦";   // hypothesis index, or ✦ for a random π_ref action
              b.style.outline = `2px ${a.ref ? "dashed" : "solid"} ${col}`;
              b.style.outlineOffset = "-2px";
              b.style.color = col;
              b.title = a.ref ? "random reference action (π_ref)" : `H${hi + 1}'s proposed action`;
            }
          }
        }
        wrap.appendChild(b);
      }
    }
    host.appendChild(wrap);
    const note = document.getElementById("sm-boardnote");
    if (note) note.innerHTML = `Same board as <i>Play the environments</i> above — click any cell to fire. <span style="color:var(--clay)">X</span> = hit, gray = miss.`
      + (stage >= 1 ? ` Outlined cells are each hypothesis's proposed action, numbered by hypothesis.` : ``)
      + (decision ? ` <b style="color:var(--clay)">◎ = HELiX's chosen shot.</b>` : ``);
  }

  /* ---- render hypotheses list ---- */
  function renderHyps() {
    const host = document.getElementById("sm-hyps");
    host.innerHTML = "";
    if (stage < 1) { host.innerHTML = `<p class="hypo-note">Hypotheses the LLM samples will appear here, each with its best action.</p>`; return; }
    hyps.forEach((h, i) => {
      const d = el("div");
      d.style.cssText = `border-left:3px solid ${h.color}; padding:6px 0 6px 12px; margin:8px 0; font-size:13.5px;`;
      d.innerHTML = `<b style="color:${h.color}">H${i + 1}.</b> ${h.text}`;
      host.appendChild(d);
    });
    const refs = actions.filter(a => a.ref).map(a => a.name).join(", ");
    if (refs) host.insertAdjacentHTML("beforeend",
      `<p class="hypo-note" style="margin-top:10px;"><i>Optional</i> $\\pi_{\\text{ref}}$ extension also samples random actions as a baseline: <b>${refs}</b></p>`);
  }

  // Two markers only: GREEN = this hypothesis's favorite action (row argmax);
  // everything else is a neutral tile. (Red ring for the final pick is added in
  // renderMatrix.) The number is always shown; colour no longer encodes magnitude.
  function paintCell(td, v, adv, isArg) {
    td.textContent = adv ? (v > 0 ? "+" : (v < 0 ? "−" : "")) + Math.abs(v).toFixed(2) : v.toFixed(2);
    if (isArg) { td.style.background = "var(--moss)"; td.style.color = "#fff"; }
    else { td.style.background = "var(--bg-sunk)"; td.style.color = "var(--ink-soft)"; }
  }

  /* ---- render matrix (raw scores, or advantages once re-scored) ---- */
  function renderMatrix() {
    const host = document.getElementById("sm-matrixwrap");
    host.innerHTML = "";
    if (stage < 2) return;

    const adv = rescored;            // showing advantages?
    const M = adv ? A : S;
    const argmaxSets = stage >= 3 ? rowArgmaxSets(M) : null;

    const tbl = el("table", "smatrix");
    const thead = el("tr");
    thead.appendChild(el("th", "", ""));
    actions.forEach(a => {
      const th = el("th", a.ref ? "ref" + (adv ? " dim" : "") : "act", a.name);
      thead.appendChild(th);
    });
    if (adv) thead.appendChild(el("th", "basehd", "−bₕ"));
    tbl.appendChild(thead);

    const chosenCol = [];   // cells in HELiX's chosen action column (for the box)
    hyps.forEach((h, i) => {
      const tr = el("tr");
      const hcell = el("td", "hyp");
      hcell.innerHTML = `H${i + 1}`;
      hcell.style.color = h.color;
      tr.appendChild(hcell);
      actions.forEach((a, j) => {
        const td = el("td", "sc filled");
        const isArg = !!(argmaxSets && argmaxSets[i].has(j));
        // in the advantage view, reference columns keep their RAW belief (no +/− sign)
        paintCell(td, M[i][j], adv && !a.ref, isArg);
        if (adv && a.ref) td.classList.add("dim");
        // box only the argmax (green) cells of the chosen column: a single cell
        // when one hypothesis picks it, the whole column only when ≥2 agree.
        if (decision && decision.actIdx === j && isArg && !a.ref) chosenCol.push(td);
        tr.appendChild(td);
      });
      if (adv) {
        const bc = el("td", "base");
        bc.textContent = (baseline[i] >= 0.005 ? "−" : "") + baseline[i].toFixed(2);
        bc.title = "baseline bₕ = average belief on the random π_ref actions";
        tr.appendChild(bc);
      }
      tbl.appendChild(tr);
    });
    host.appendChild(tbl);
    if (decision && chosenCol.length) drawColBox(host, chosenCol);

    const legend = `<b style="color:var(--moss)">Green</b> = each hypothesis's favorite action (row argmax)${decision ? `; <b style="color:var(--clay)">red box</b> = HELiX's final pick (one cell, or a column when hypotheses agree)` : ``}.`;
    if (adv) {
      host.insertAdjacentHTML("beforeend",
        `<p class="hypo-note" style="margin-top:8px;">Now showing <b>advantages</b> = score − bₕ, applied to the <b>real action columns only</b>.
        bₕ is each hypothesis's average belief on the random <span style="font-style:italic">π_ref</span> columns (dashed) — those columns
        keep their <i>raw</i> belief (no subtraction), since they're what bₕ averages. Spread-out rows shrink toward 0; a
        <em>concentrated</em> hypothesis keeps a high advantage. <span style="font-style:italic">All values shown to 2 decimals.</span><br>${legend}</p>`);
    } else if (stage >= 3) {
      host.insertAdjacentHTML("beforeend",
        `<p class="hypo-note" style="margin-top:8px;">${legend}${usePiref ? " Dashed columns are the optional random π_ref actions." : ""} <span style="font-style:italic">Scores shown to 2 decimals.</span></p>`);
    }
  }

  // Draw ONE rounded red box around the chosen action's column (instead of a
  // per-cell ring, which bleeds when several cells stack). Measured from the live
  // layout; skipped under the headless test stub (no getBoundingClientRect).
  function drawColBox(host, cells) {
    if (!cells.length || typeof cells[0].getBoundingClientRect !== "function") return;
    const w = host.getBoundingClientRect();
    let L = Infinity, T = Infinity, R = -Infinity, B = -Infinity;
    cells.forEach(c => {
      const r = c.getBoundingClientRect();
      L = Math.min(L, r.left); T = Math.min(T, r.top);
      R = Math.max(R, r.right); B = Math.max(B, r.bottom);
    });
    const box = el("div", "smatrix-colbox");
    box.style.left = (L - w.left + host.scrollLeft - 4) + "px";
    box.style.top = (T - w.top + host.scrollTop - 4) + "px";
    box.style.width = (R - L + 8) + "px";
    box.style.height = (B - T + 8) + "px";
    host.appendChild(box);
  }

  // row argmax over the REAL action columns only (reference columns are a baseline, never a pick)
  function rowArgmaxSets(M) {
    return M.map(row => {
      let mx = -Infinity;
      actions.forEach((a, j) => { if (!a.ref && row[j] > mx) mx = row[j]; });
      const set = new Set();
      actions.forEach((a, j) => { if (!a.ref && Math.abs(row[j] - mx) < 1e-9) set.add(j); });
      return set;
    });
  }

  // Most-optimistic action: the highest score over all (hypothesis, non-random
  // action) cells. When several cells tie at the top — which is the common case,
  // since each hypothesis rates its own favorite ~1.0 — HELiX breaks the tie at
  // RANDOM. Returns {h, a, score, tie, nTie} where nTie = # of tied top actions.
  function twoTier(M) {
    let best = -Infinity;
    M.forEach((row) => actions.forEach((a, j) => { if (!a.ref && row[j] > best + 1e-9) best = row[j]; }));
    const top = [];
    M.forEach((row, i) => actions.forEach((a, j) => { if (!a.ref && Math.abs(row[j] - best) < 1e-9) top.push({ h: i, a: j }); }));
    const tieActions = [...new Set(top.map(t => t.a))];           // distinct actions in the tie pool
    const a = tieActions[(Math.random() * tieActions.length) | 0]; // random tie-break over actions
    const h = top.find(t => t.a === a).h;
    return { h, a, score: best, tie: tieActions.length > 1, nTie: tieActions.length, tieActions };
  }

  /* ---- the decision ---- */
  function decide() {
    const sets = rowArgmaxSets(S);
    let inter = [...sets[0]];
    for (let i = 1; i < sets.length; i++) inter = inter.filter(j => sets[i].has(j));

    if (inter.length > 0) {
      // consensus -> exploit; tie-break: prefer non-ref, earliest
      inter.sort((a, b) => (actions[a].ref - actions[b].ref) || (a - b));
      decision = { type: "exploit", actIdx: inter[0] };
      return;
    }

    // no consensus -> explore. Always compute the raw ("core HELiX") pick.
    const raw = twoTier(S);
    if (usePiref) {
      if (!rescored) doRescore();
      const ref = twoTier(A);
      decision = {
        type: "explore", piref: true,
        rawIdx: raw.a, rawHyp: raw.h, rawTie: raw.tieActions,
        actIdx: ref.a, hypIdx: ref.h, adv: ref.score, refTie: ref.tieActions
      };
    } else {
      decision = { type: "explore", piref: false, actIdx: raw.a, hypIdx: raw.h, score: raw.score, tieActions: raw.tieActions };
    }
  }

  function renderDecision() {
    const host = document.getElementById("sm-decision");
    if (!decision) { host.style.display = "none"; return; }
    host.style.display = "block";
    const a = actions[decision.actIdx];

    if (decision.type === "exploit") {
      host.className = "decision show-exploit";
      const pirefNote = usePiref
        ? ` <span class="hypo-note">(π_ref re-scoring isn't needed here — it only changes the explore step, and there's already consensus.)</span>`
        : "";
      host.innerHTML = `<span class="pill exploit">EXPLOIT · consensus</span><br><br>
        Every hypothesis's row-argmax shares the action <b>${a.name}</b> — the intersection is non-empty.
        By the minimax ⇒ common-optimum lemma, it is simultaneously best for <i>all</i> surviving
        hypotheses, so HELiX commits without further exploration. <b>Fire ${a.name}.</b>${pirefNote}`;
    } else if (!decision.piref) {
      host.className = "decision show-explore";
      const nm = idxs => idxs.map(j => actions[j].name).join(", ");
      const pool = decision.tieActions;
      const pick = pool.length > 1
        ? `The top score (${decision.score.toFixed(2)}) is a tie, so the action is a <b>random&#8209;tie&#8209;break(${nm(pool)})</b> — this turn, <b>${a.name}</b>.`
        : `The single most optimistic action is <b>${a.name}</b> (under H${decision.hypIdx + 1}).`;
      host.innerHTML = `<span class="pill explore">EXPLORE · no consensus</span><br><br>
        The row-argmaxes don't share an action — the hypotheses disagree, so core HELiX <i>explores</i>.
        ${pick} <b>Fire ${a.name} to gather information.</b>
        <span class="hypo-note">Turn on <b>π_ref re-scoring</b> to see the optional advantage tie-breaker.</span>`;
    } else {
      host.className = "decision show-explore";
      const raw = actions[decision.rawIdx];
      const nm = idxs => idxs.map(j => actions[j].name).join(", ");
      // the hypothesis re-scoring demotes most = the one with the most belief on random cells
      let spreadIdx = 0; baseline.forEach((b, i) => { if (b > baseline[spreadIdx]) spreadIdx = i; });
      const rawPool = decision.rawTie, refPool = decision.refTie;
      const narrowed = refPool.length < rawPool.length;
      const rawLine = rawPool.length > 1
        ? `the top raw score ties, so the pick is <b>random&#8209;tie&#8209;break(${nm(rawPool)})</b> — this turn <b>${raw.name}</b>.`
        : `the most optimistic action is <b>${raw.name}</b>.`;
      const refLine = !narrowed
        ? `the tie-break pool is unchanged here, though the spread-out rows still flatten — <b>random&#8209;tie&#8209;break(${nm(refPool)})</b>, this turn <b>${a.name}</b>.`
        : refPool.length > 1
          ? `the spread-out hypotheses (like <b>H${spreadIdx + 1}</b>) leak probability onto those random cells and drop out, shrinking the pick to <b>random&#8209;tie&#8209;break(${nm(refPool)})</b> — this turn <b>${a.name}</b>.`
          : `the spread-out hypotheses drop out, leaving only <b>${a.name}</b> with a top advantage.`;
      host.innerHTML = `<span class="pill explore">EXPLORE · no consensus</span><br><br>
        <b>Without π_ref</b> (core HELiX): ${rawLine}<br>
        <b>With π_ref re-scoring</b> (optional): subtract each hypothesis's <i>average belief on the random
        π_ref cells</i> (its bₕ); ${refLine}<br><br>
        Re-scoring shrinks the tie-break pool to hypotheses that actually pin down a location, over vague
        spread-out guesses.`;
    }
    if (window.MathJax && MathJax.typesetPromise) MathJax.typesetPromise([host]);
  }

  /* ---- stage machine (adaptive: the explore+π_ref path has an extra re-score step) ---- */
  function stepKind(s) {
    if (s === 1) return "sample";
    if (s === 2) return "build";
    if (s === 3) return "argmax";
    if (s === 4) return needsRescore ? "rescore" : "decide";
    if (s === 5) return "decide";
    return "none";
  }
  const terminalStage = () => (needsRescore ? 5 : 4);

  function narration() {
    if (board && board.over) return `🏁 <b>Game over.</b> Press <b>New game</b> to play another round.`;
    switch (stepKind(stage)) {
      case "none": return `This is your <b>Battleship board</b> from above — fire a shot on either board and both update. Press <b>Sample hypotheses</b> to see what HELiX would do here.`;
      case "sample": return `<b>Stage 1 — Sample.</b> The LLM proposes feedback-consistent hypotheses, each with a best action${usePiref ? `, plus a few <i>optional</i> random π_ref actions (dashed) as a baseline` : ``}. Next: score every action under every hypothesis.`;
      case "build": return `<b>Stage 2 — Build the score matrix.</b> $R_{\\text{LLM}}(\\eta, a)$ rates each action under each hypothesis. An action proposed <i>for</i> a hypothesis scores high <i>under</i> it.`;
      case "argmax": return `<b>Stage 3 — Row argmaxes.</b> Mark each hypothesis's highest-scoring action. If they share one → exploit; if not → explore.`;
      case "rescore": return `<b>Stage 4 — Re-score against π_ref (optional).</b> Subtract each hypothesis's baseline $b_h$ (its average belief on the random cells). Raw scores become <i>advantages</i> — watch the spread-out rows shrink toward 0.`;
      case "decide": return `<b>Stage ${stage} — Decide.</b> Read the verdict below, then let HELiX take the shot — it fires on the board and you continue from the new position.`;
    }
  }
  function btnLabel() {
    if (board && board.over) return "↺ New game";
    if (stage === 0) return "Sample hypotheses";
    if (stage === terminalStage()) {
      const pick = decision && actions[decision.actIdx];
      return pick ? `◎ Fire HELiX's pick: ${pick.name} →` : "↺ Start over";
    }
    switch (stepKind(stage)) {
      case "sample": return "Build score matrix →";
      case "build": return "Mark row argmaxes →";
      case "argmax": return needsRescore ? "Re-score with π_ref →" : "Make the decision →";
      case "rescore": return "Make the decision →";
      default: return "Next →";
    }
  }

  function advance() {
    if (board && board.over) { Battleship.reset(); return; }   // game over -> new game
    if (stage === terminalStage()) {
      // let HELiX actually take its shot: fire the chosen cell on the shared board.
      // Battleship.fire() then re-syncs and resets the reasoning for the next move.
      const pick = decision && actions[decision.actIdx];
      if (pick) Battleship.fire(pick.rc[0], pick.rc[1]);
      else { resetState(); renderAll(); }
      return;
    }
    stage++;
    if (stage === 1) sampleHypotheses();
    if (stage === 2) buildMatrix();                 // sets consensusFound, needsRescore
    if (stage === 4 && needsRescore) doRescore();   // show advantage matrix
    if ((stage === 4 && !needsRescore) || stage === 5) decide();
    renderAll();
  }

  function renderAll() {
    document.getElementById("sm-narration").innerHTML = narration();
    document.getElementById("sm-step").textContent = btnLabel();
    renderBoard(); renderHyps(); renderMatrix(); renderDecision();
    const n = document.getElementById("sm-narration");
    if (window.MathJax && MathJax.typesetPromise) MathJax.typesetPromise([n, document.getElementById("sm-hyps")]);
  }

  function resetState() {
    hyps = []; actions = []; S = []; A = []; baseline = [];
    stage = 0; decision = null; rescored = false; consensusFound = false; needsRescore = false;
  }

  function setPiref(on) {
    usePiref = on;
    const btn = document.getElementById("sm-piref");
    btn.textContent = `+ π_ref re-scoring: ${on ? "on" : "off"}`;
    btn.setAttribute("aria-pressed", on ? "true" : "false");
    resetState();                 // re-sample on the SAME board so users can compare
    renderAll();
  }
  function togglePiref() { setPiref(!usePiref); }

  // The sandbox board IS the Battleship game board (kept in sync by Battleship).
  // Called on every shot / new game; rebuilds the ship context and resets the
  // reasoning to stage 0 (any board change invalidates the old hypotheses).
  function syncFromBattleship(state) {
    board = { shots: state.shots, turns: state.turns };
    // reason about the active (non-sunk) hit streak only
    const hits = (state.activeHits && state.activeHits.length) ? state.activeHits : (() => {
      const h = []; for (let r = 0; r < N; r++) for (let c = 0; c < N; c++) if (state.shots[r][c] === 2) h.push([r, c]); return h;
    })();
    let horiz = true;
    if (hits.length >= 2) horiz = hits.every(h => h[0] === hits[0][0]);
    const remaining = (state.remaining && state.remaining.length) ? state.remaining : SHIP_LENS.slice();
    board.hitLine = { horiz };
    board.activeHits = hits;
    board.sunk = state.sunk || [];
    board.remaining = remaining;
    board.minRem = Math.min(...remaining);
    board.streakLen = hits.length || 1;
    board.over = !!state.over;
    board.sunkCells = new Set((state.sunkCells || []).map(c => c[0] + "," + c[1]));
    resetState();
    renderAll();
  }

  return { advance, togglePiref, syncFromBattleship };
})();

/* ======================================================================
   WIRING
   ====================================================================== */
window.addEventListener("DOMContentLoaded", () => {
  // Battleship
  Battleship.reset();
  document.getElementById("bs-reset").onclick = Battleship.reset;
  document.getElementById("bs-helix").onclick = Battleship.suggest;

  // Minesweeper — only if its widget is present (it may be commented out)
  if (document.getElementById("ms-board")) {
    Minesweeper.reset();
    document.getElementById("ms-reset").onclick = Minesweeper.reset;
    document.getElementById("ms-flag").onclick = Minesweeper.toggleFlagMode;
  }

  // Score matrix — board is the shared Battleship board (synced above)
  document.getElementById("sm-step").onclick = ScoreMatrix.advance;
  document.getElementById("sm-piref").onclick = ScoreMatrix.togglePiref;
  document.getElementById("sm-new").onclick = Battleship.reset;   // "New game" resets both boards
});
