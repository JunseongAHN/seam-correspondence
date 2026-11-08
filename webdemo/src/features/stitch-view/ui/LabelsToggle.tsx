/** Show "index · arc length" on every edge. */
export default function LabelsToggle({ labels, onChange }: {
  labels: boolean;
  onChange: (labels: boolean) => void;
}) {
  return (
          <label className="note" style={{ userSelect: "none" }}>
            <input type="checkbox" checked={labels}
                   onChange={(e) => onChange(e.target.checked)} />{" "}
            edge index and length
          </label>
  );
}
