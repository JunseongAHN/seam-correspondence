import { MORE } from "../logic";
import shared from "@/styles/shared.module.css";
import s from "./MoreGarments.module.css";

/* The five are GarmentCodeData too, so the row appears once example 2 is chosen
   and stays while one of them is loaded. It sits below the drop zone rather than
   in it: the zone is for getting a pattern in, and this is a whole second reading
   -- five scores and the sentence explaining how they were picked. */
export default function MoreGarments({ busy, name, onPick }: {
  busy: boolean;
  /** the loaded file's name, to mark which garment is open */
  name: string;
  onPick: (id: string) => void;
}) {
  return (
      <div className={`${s.tryit} ${s.more}`}>
        more held-out garments, from worst to best —{" "}
        {MORE.map((g, i) => (
          <span key={g.id}>
            {i > 0 && <span className={s.sep}>·</span>}
            <button className={`${s.link} ${name.startsWith(g.id) ? s.on : ""}`}
                    disabled={busy} title={g.what}
                    onClick={() => onPick(g.id)}>
              F1 {g.f1}
            </button>
          </span>
        ))}
        <span className={shared.note} style={{ marginLeft: 10 }}>
          picked by scoring a sample of the test split with the model this page runs, then
          taking a spread — not a highlight reel. Each opens beside{" "}
          <strong className={shared.gt}>its ground-truth drape, in red</strong>.
        </span>
      </div>
  );
}
