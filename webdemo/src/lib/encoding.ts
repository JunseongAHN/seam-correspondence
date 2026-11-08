/* Which curvature encoding the pipeline uses.  Its own file because both sides need it:
   the DXF reader skips the curve refit under "sagitta", and the feature builder lays out
   dims 7..17 by it -- and a reader importing it from the feature builder would make
   pattern parsing depend on inference. */
export type CurvatureEncoding = "tagged" | "sagitta";
/** Must match the checkpoint the ONNX model was exported from. */
export const ENCODING: CurvatureEncoding = "sagitta";
