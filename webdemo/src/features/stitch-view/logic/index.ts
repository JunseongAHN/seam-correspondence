export { placePanels, type Placed } from "./render";
export { midpointsByNode, patternGeometry, svgFrame } from "./geometry";
export { layoutLabels, type LabelEdge, type LabelObstacle } from "./labelLayout";
export { edgeStyle, focusNodes, isLit, nodeIndex, pairGroups, pairPartners, touchesFocus,
         stitchIndexByEdge } from "./pairs";
export { clientToUser, dragView, strokeScale, unitsPerPixel, zoomAt,
         type Drag, type View } from "./viewport";
