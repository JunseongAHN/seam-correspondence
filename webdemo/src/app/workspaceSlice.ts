/* State more than one feature reads or changes: the loaded result, what it was loaded as,
   whether a load is running, the model's status and the page's error.

   The result itself is split.  What is small and plain -- its name, sizes, timing and the
   predicted and ground-truth pairs -- is here; the parsed pattern, the feature tensor and
   the model's scores are in adapters/resultCache, and `result.id` finds them. */
import { createAsyncThunk, createSelector, createSlice, type PayloadAction } from "@reduxjs/toolkit";
import { addPair, removePair, toPairList, toPairSet } from "@/adapters/pairs";
import { dropResult, putResult, retainOnly } from "@/adapters/resultCache";
import { gtPairs, modelLoaded, patternToTensors, predict } from "@/core-algorithm/inference";
import { annotationClick, gtFromDoc } from "@/features/annotation/logic";
import { scoreStats } from "@/features/evaluation/logic";
import { parseInput, type ExampleKey } from "@/features/input/logic";

export type ModelStatus = "idle" | "loading" | "ready" | "error";

/** A loaded result as the store keeps it. Pairs are "a-b" node pairs, in the order made. */
export type ResultMeta = {
  id: string; name: string; panels: number; edges: number; ms: number;
  pred: string[]; gt: string[];
};

/** What the 3D pane needs to know about how the result was opened.  input owns the
    example catalogue, so it passes these along rather than the store looking them up. */
export type PaneSource = {
  exampleRight: "sim" | "drape" | null;
  moreSim: { file: string; expectS: number } | null;
};

export type WorkspaceState = {
  result: ResultMeta | null;
  /** the last file loaded; kept when a load fails, as the "on" marker always was */
  name: string;
  example: ExampleKey | null;
  more: string | null;          // which held-out garment
  pane: PaneSource;
  busy: boolean;
  modelStatus: ModelStatus;
  error: string | null;
  /** the newest openPattern; an older one that finishes later is ignored */
  latest: string | null;
  /* Drawing a ground truth by hand: annotation's toolbar changes these, and the 2D view
     both draws them and takes the clicks that make them. */
  annotate: boolean;
  userPairs: string[];          // in the order they were sewn: it picks their colours
  selected: number | null;      // the first edge of a pair being made
};

const initialState: WorkspaceState = {
  result: null, name: "", example: null, more: null,
  pane: { exampleRight: null, moreSim: null },
  busy: false, modelStatus: "idle", error: null, latest: null,
  annotate: false, userPairs: [], selected: null,
};

/** A file the reader chose, or a URL input resolved for an example. */
export type OpenArg =
  | { source: "file"; file: File }
  | ({ source: "url"; url: string; fileName: string; gtUrl: string | null;
       example: ExampleKey | null; more: string | null } & PaneSource);

type Opened = { result: ResultMeta; example: ExampleKey | null; more: string | null; pane: PaneSource };
/* Two ways a load fails, kept apart because the page has always treated them differently:
   a download that fails leaves the previous result on screen; a pattern that cannot be
   read or run clears it.  `model` is where the model ended up, or null if it was not
   reached. */
type Failed = { stage: "fetch" | "load"; message: string; model: ModelStatus | null };

const message = (e: any) => String(e?.message ?? e);

/** One pipeline for every input: text -> Pattern -> features -> ONNX -> pairs.

    Loads can overlap -- the drop zone takes a file while an example is still running --
    and the slower one used to win.  Only the newest is applied now; see `latest`. */
export const openPattern = createAsyncThunk<Opened, OpenArg,
    { rejectValue: Failed; state: { workspace: WorkspaceState } }>(
  "workspace/openPattern",
  async (arg, { dispatch, getState, rejectWithValue, requestId }) => {
    const isLatest = () => getState().workspace.latest === requestId;
    let text: string, fileName: string;
    try {
      if (arg.source === "file") { text = await arg.file.text(); fileName = arg.file.name; }
      else {
        const r = await fetch(arg.url);
        if (!r.ok) throw new Error(`${arg.url}: HTTP ${r.status}`);
        text = await r.text(); fileName = arg.fileName;
      }
    } catch (e) { return rejectWithValue({ stage: "fetch", message: message(e), model: null }); }

    let triedModel = false;
    try {
      const { pattern, isDxf } = parseInput(text, fileName);
      const t = patternToTensors(pattern);
      // the first prediction also fetches the model, which takes far longer than running it
      if (!modelLoaded()) { triedModel = true; dispatch(modelLoading()); }
      const { pairs, ms, scores } = await predict(t);
      let gt = isDxf ? new Set<string>() : gtPairs(pattern, t.keys);
      const gtUrl = arg.source === "url" ? arg.gtUrl : null;
      if (gtUrl) {
        const g = await fetch(gtUrl);
        if (!g.ok) throw new Error(`${gtUrl}: HTTP ${g.status}`);
        gt = gtFromDoc(await g.json(), t.keys);
      }
      const id = putResult({ pattern, keys: t.keys, x: t.x, scores });
      // a stale load must not evict the result the page is showing
      if (isLatest()) retainOnly(id); else dropResult(id);
      return {
        result: { id, name: fileName, panels: pattern.panels.length, edges: t.M, ms,
                  pred: toPairList(pairs), gt: toPairList(gt) },
        example: arg.source === "url" ? arg.example : null,
        more: arg.source === "url" ? arg.more : null,
        pane: arg.source === "url" ? { exampleRight: arg.exampleRight, moreSim: arg.moreSim }
                                   : { exampleRight: null, moreSim: null },
      };
    } catch (e) {
      if (isLatest()) retainOnly(null);
      return rejectWithValue({ stage: "load", message: message(e),
                               model: modelLoaded() ? "ready" : triedModel ? "error" : null });
    }
  });

/** A new result is on the page.  Slices that should react to that listen for this. */
export const resultLoaded = openPattern.fulfilled;

const workspaceSlice = createSlice({
  name: "workspace",
  initialState,
  reducers: {
    modelLoading(state) { state.modelStatus = "loading"; },
    errorSet(state, action: PayloadAction<string | null>) { state.error = action.payload; },
    annotateToggled(state) { state.annotate = !state.annotate; state.selected = null; },
    /** Click one edge then another to sew them together; click a paired edge to undo it. */
    edgeClicked(state, { payload: n }: PayloadAction<number>) {
      const { selected, change } = annotationClick(state.selected, toPairSet(state.userPairs), n);
      if (change) state.userPairs = change.remove ? removePair(state.userPairs, change.key)
                                                  : addPair(state.userPairs, change.key);
      state.selected = selected;
    },
    pairsCleared(state) { state.userPairs = []; state.selected = null; },
    /** A saved ground truth was loaded in place of the hand-drawn one. */
    pairsLoaded(state, { payload }: PayloadAction<string[]>) {
      state.userPairs = payload; state.error = null;
    },
  },
  extraReducers: (b) => b
    .addCase(openPattern.pending, (state, { meta }) => {
      state.latest = meta.requestId; state.busy = true; state.error = null;
    })
    /* There is one model, whichever load fetched it, so a stale load still reports that it
       is now in memory.  Nothing else of a stale load is applied -- not even a model
       failure, which would hide "loading the model" while the newer load is still at it. */
    .addCase(openPattern.fulfilled, (state, { payload, meta }) => {
      state.modelStatus = "ready";
      if (meta.requestId !== state.latest) return;
      state.result = payload.result;
      state.name = payload.result.name;
      state.example = payload.example;
      state.more = payload.more;
      state.pane = payload.pane;
      state.busy = false;
    })
    .addCase(openPattern.rejected, (state, { payload, error, meta }) => {
      const stale = meta.requestId !== state.latest;
      if (payload?.model && (!stale || payload.model === "ready")) state.modelStatus = payload.model;
      if (stale) return;
      state.busy = false;
      state.error = payload?.message ?? String(error.message);
      if (payload?.stage === "load") { state.result = null; state.example = null; state.more = null; }
    }),
});

export const { modelLoading, errorSet, annotateToggled, edgeClicked, pairsCleared, pairsLoaded } =
  workspaceSlice.actions;

/* Derived values more than one feature reads.  Typed against the slice alone, not the
   store, so this module does not import store.ts and close a cycle. */
type WithWorkspace = { workspace: WorkspaceState };
const EMPTY: string[] = [];
const selectResult = (s: WithWorkspace) => s.workspace.result;
const selectUserPairList = (s: WithWorkspace) => s.workspace.userPairs;

export const selectPred = createSelector([(s: WithWorkspace) => s.workspace.result?.pred ?? EMPTY], toPairSet);
export const selectUserPairs = createSelector([selectUserPairList], toPairSet);
/** The ground truth the prediction is scored against: hand-drawn outranks the file's, so a
    DXF becomes scorable. */
export const selectTruth = createSelector(
  [selectUserPairList, (s: WithWorkspace) => s.workspace.result?.gt ?? EMPTY],
  (mine, file) => toPairSet(mine.length ? mine : file));
export const selectScored = createSelector([selectResult, selectTruth], (r, truth) => !!r && truth.size > 0);
export const selectStats = createSelector([selectResult, selectPred, selectTruth],
  (r, pred, truth) => (r ? scoreStats(pred, truth) : null));
export default workspaceSlice.reducer;
