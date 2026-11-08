/* The ground-truth drape: a garment already assembled, shipped as geometry.

   Nothing here is solved in the browser.  The solve ran once on the host with the
   native build of the same solver the demo carries, from the ground-truth seams, and
   only its result travels -- so the reference shape is on screen immediately instead of
   after a ten-second freeze.

   Layout (little-endian), written by simcpp/pack_gt_drape.py:
     "GTDRAPE1", int32 n, M, NC, NS,
     pos (n,3) f32, faces (M,3) i32, cyl (NC,7) f32, sph (NS,4) f32 */
const B = import.meta.env.BASE_URL;

export type Drape = {
  n: number; M: number;
  pos: Float32Array;      // (n,3) solved positions, cm
  faces: Uint32Array;     // (M,3)
  cyl: Float32Array;      // (NC,7) capsule p0, p1, r -- the solver's body proxy
  sph: Float32Array;      // (NS,4) centre, r
};

export async function loadDrape(id: string): Promise<Drape> {
  const url = `${B}sim/${id}_drape.bin`;
  const r = await fetch(url);
  if (!r.ok) throw new Error(`${url}: HTTP ${r.status}`);
  const buf = await r.arrayBuffer();
  const magic = new TextDecoder().decode(new Uint8Array(buf, 0, 8));
  if (magic !== "GTDRAPE1") throw new Error(`${url}: not a drape file (${magic})`);
  const h = new Int32Array(buf, 8, 4);
  const [n, M, NC, NS] = [h[0], h[1], h[2], h[3]];
  let o = 8 + 16;
  const pos = new Float32Array(buf.slice(o, o + n * 3 * 4)); o += n * 3 * 4;
  const faces = new Uint32Array(buf.slice(o, o + M * 3 * 4)); o += M * 3 * 4;
  const cyl = new Float32Array(buf.slice(o, o + NC * 7 * 4)); o += NC * 7 * 4;
  const sph = new Float32Array(buf.slice(o, o + NS * 4 * 4));
  return { n, M, pos, faces, cyl, sph };
}
