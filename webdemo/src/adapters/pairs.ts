/* Stitch pairs ("a-b", node indices, a < b) as the store holds them: a string[] in the order
   the pairs were made, not a Set, which is not serializable.  Not sorted: the 2D view numbers
   correspondence groups -- and so picks their colours -- in that order, and sorting would
   recolour the pattern and shift colours while a ground truth is being drawn. */

/** A Set (or any list) of pairs as the store's list: first occurrence wins, order kept. */
export function toPairList(pairs: Iterable<string>): string[] {
  return [...new Set(pairs)];
}

/** The store's list as a Set, for the code that looks pairs up. */
export function toPairSet(list: readonly string[]): Set<string> {
  return new Set(list);
}

/** What Set.add does to the list: appended if new, unchanged otherwise. */
export function addPair(list: readonly string[], key: string): string[] {
  return list.includes(key) ? [...list] : [...list, key];
}

/** What Set.delete does to the list. */
export function removePair(list: readonly string[], key: string): string[] {
  return list.filter((k) => k !== key);
}
