# 分析与思考

## 基础操作提取

从20道参考题中，我识别出以下基础操作类别：

1. **空间移动与对齐**：平移对象到边界/锚点（#3, #13）、重力/掉落（#20）、旋转/镜像对齐（#18）
2. **缩放与印章**：用tile放大像素（#4）、用glyph替换控制点（#10）、minimap到实际尺寸映射（#12）
3. **路径与图论**：最短路径（#2）、沿corridor前进并转弯（#17）、沿guide像素路径查找目标（#7）
4. **拼接与排序**：边缘匹配排序（#6）、拼合为矩形（#1, #11）
5. **填充与绘制**：Chebyshev距离环填充（#15）、三角形填充（#8）、容器填充（#20）、toggle射线（#16）
6. **颜色编码与标记**：颜色作为分组标签（#14）、颜色匹配目标方向（#18）、颜色匹配锚定边界（#13）
7. **信息复制与对称**：点反射复制（#9）、sprite复制到marker交叉点（#5）
8. **长度/对齐规范化**：线段对齐到参考线（#19）

## 非显而易见的串联方式

- **计数结果作为几何参数**：#8中控制点数量变成三角形边长。类似地，某种属性的计数可以决定输出的形状维度、重复次数、偏移量等。
- **拓扑特征作为选择条件**：#20中U形容器的"封闭性"决定填充行为。结构的拓扑属性（封闭/开放、连通分量数、洞的数量）可以作为规则分支条件。
- **相对位置编码绝对信息**：#15中标记的Chebyshev距离编码颜色分配；#7中路径颜色由相邻guide像素决定。
- **对象间的隐含对应关系**：不通过颜色匹配，而通过形状互补（#11矩形补全）、面积比例、或空间对称性建立对应。
- **输出尺寸由内容决定**：#1中像素总数决定frame尺寸；#4中控制sprite尺寸乘以tile尺寸决定输出维度。

## 寻找新的因果关系

- 对象的**面积排名**可以决定其变换方式（而非颜色或位置）
- 对象的**形状类别**（如L形/T形/直线）可以编码操作指令
- **连通路径的长度**可以决定填充颜色的选择
- 对象之间的**相对方位**可以决定组合方式
- **边界接触模式**（接触哪条边、接触几条边）可以编码层级或优先级信息
- **对象内部的孔洞数量**可以编码排序优先级或选择条件
- **对象轮廓的凹凸性**可以决定变换类型（凸对象做一种操作，凹对象做另一种）

---

# 生成新谜题

<new_puzzle>
Flood-fill each enclosed room with the color of the single door pixel that seals it.

Input structure:
- bg = black background.
- Grey pixels form a connected maze-like wall structure on the grid, creating multiple enclosed rectangular rooms. Each room is a rectangular region completely surrounded by grey walls on all four sides, except for exactly one gap of exactly one pixel on one wall edge.
- Each gap pixel is filled by a single colored pixel (non-grey, non-black) -- this is the "door" pixel. Different rooms have different door colors drawn from: red, blue, green, yellow, orange, magenta, azure, maroon.
- The interior of each room (the black rectangular area bounded by grey walls and the door pixel) is entirely black.
- Rooms do not share walls; each room's wall structure is self-contained, though rooms may be adjacent with a double-thickness wall between them.

Rule:
- For each enclosed room, identify its door pixel: the unique non-grey, non-black pixel that sits in the single gap of its wall boundary.
- Fill the entire black interior of that room with the door pixel's color.
- The grey walls, door pixels, and any black area outside rooms all remain unchanged.
- Each door belongs to exactly one room.

Key insight:
The door pixel is visually part of the wall boundary and appears to be a wall pixel at first glance -- solvers must realize it is not a wall pixel but a color label for the room it seals. Identifying "enclosed" requires understanding that the grey walls plus the door pixel together form a complete boundary, and the door's color propagates inward to fill the entire interior.
</new_puzzle>

<new_puzzle>
Reconstruct the hidden complete shape by un-rotating and overlaying partial views from multiple panels, each rotated by the angle encoded in its corner marker color.

Input structure:
- bg = black background.
- The grid is divided into equal square panels by single-pixel-wide grey separator lines (both horizontal and vertical), forming a regular 2x2 arrangement of four panels, each of size NxN (where N is between 7 and 12).
- Each panel contains: (a) a partial foreground shape drawn in blue pixels, and (b) exactly one colored marker pixel placed in one specific corner of the panel (the same relative corner position across all panels, e.g., all in top-left). The marker color encodes a rotation amount: red = 0 degrees (no rotation), green = 90 degrees clockwise, yellow = 180 degrees, magenta = 270 degrees clockwise.
- Each partial shape, when rotated back (counter-rotated) by the angle its marker specifies, reveals a fragment of the same underlying hidden shape. Different panels show different non-overlapping fragments.

Rule:
- For each panel, take only the blue pixels (ignoring the marker pixel).
- Rotate the blue pixel pattern by the inverse of the marker-specified angle around the panel's center: red panels stay unchanged, green panels rotate 90 degrees counter-clockwise, yellow panels rotate 180 degrees, magenta panels rotate 270 degrees counter-clockwise.
- Overlay (union) all four de-rotated patterns onto a single NxN output grid.
- Any cell covered by blue in at least one de-rotated view is blue in the output; all others are black.
- The output grid is a single panel of size NxN (separators and markers removed).

Key insight:
Each panel is a rotated partial exposure of the same underlying shape -- the corner marker color is not decorative but an angular instruction. The solver must discover the color-to-rotation mapping by noticing that fragments only align into a coherent single shape under the correct inverse rotations, and that no individual panel contains the whole shape.
</new_puzzle>

<new_puzzle>
Connect same-colored pixel pairs with L-shaped paths, choosing the bend whose corner pixel is closer to the grid center.

Input structure:
- bg = black background.
- The grid (between 10x10 and 20x20) contains several pairs of same-colored single pixels. Each non-black color appears exactly twice on the grid. Colors used include red, blue, green, yellow, orange, magenta, azure, and maroon (at least 3 pairs, at most 7).
- The two pixels of each pair are never on the same row or same column (they always form a strict rectangular offset, requiring an L-shaped connection).
- No two pixels of different colors share a row or column.

Rule:
- For each same-colored pair at positions (r1, c1) and (r2, c2), consider the two possible L-shaped one-bend paths connecting them: Path A goes horizontal-first from (r1,c1) to (r1,c2), then vertical to (r2,c2), with bend at (r1,c2). Path B goes vertical-first from (r1,c1) to (r2,c1), then horizontal to (r2,c2), with bend at (r2,c1).
- Choose the path whose bend pixel has smaller Chebyshev distance to the grid center (center cell = (H//2, W//2) for a grid of height H, width W). If both bends have equal Chebyshev distance, choose Path A (horizontal-first).
- Draw the chosen L-path in that pair's color: color all pixels along the horizontal segment and the vertical segment, including both endpoints and the bend.
- The puzzle inputs guarantee that under the correct rule, no two paths of different colors share any pixel.

Key insight:
Both possible L-paths connect the same two endpoints and look equally valid at first -- the solver must discover that the selection criterion is spatial centrality of the bend point. This requires computing Chebyshev distances for both candidate corners to the grid center, a non-obvious geometric tiebreaker that is only discoverable by comparing multiple examples.
</new_puzzle>

<new_puzzle>
Propagate colored seeds simultaneously through a branching grey pipe network; each seed claims all reachable pipe pixels at equal speed, with collision markers at contested junctions.

Input structure:
- bg = black background.
- Grey pixels form a connected pipe network: straight segments (horizontal and vertical, each one pixel wide) that meet at junction pixels where two or more segments intersect. The network is a tree (connected, no loops/cycles).
- Several pipe endpoints (dead-end grey pixels with exactly one grey neighbor) each have a single colored pixel (non-grey, non-black) placed in the adjacent black cell touching that endpoint. These are seed pixels. Each seed is a different color, chosen from: red, blue, green, yellow, orange, magenta. There are at least 2 and at most 5 seeds.

Rule:
- All seeds simultaneously begin flooding into the grey pipe network from their adjacent endpoint.
- The flood propagates at equal speed: at each discrete step, each flood front advances one pixel along connected grey neighbors.
- When a flood front reaches a grey pixel before any other flood, that pixel is claimed and recolored to the claiming seed's color.
- When two or more flood fronts reach the same grey pixel at the same step (same pipe-distance from their respective seed entry points), that pixel becomes maroon (collision marker), and no flood passes through it -- both (or all) floods stop expanding in that direction.
- After all flooding completes, any grey pixel not reached by any flood (blocked behind collision points on all paths) remains grey.
- The colored seed pixels themselves remain in their original positions (they are not part of the pipe network and do not change).

Key insight:
The puzzle encodes a Voronoi-like territorial partition on graph distance, not Euclidean distance. The pipe structure means that two seeds might be spatially close but very far apart in pipe-distance. Solvers must simultaneously trace pipe-distance from each seed and realize that contested junctions (where path-lengths tie) produce visible maroon collision markers that block further propagation, creating uneven territory boundaries.
</new_puzzle>

<new_puzzle>
For each colored object, read the number of border edges it touches as a replication count, then tile that many copies in a horizontal row in the output.

Input structure:
- bg = black background.
- A grid (at most 25x30) contains several small monochrome 4-connected foreground objects (each a distinct non-black color, each fitting within a bounding box of at most 5x5). Each object touches at least one grid border edge (top, bottom, left, or right edge) -- meaning at least one of its pixels is on row 0, the last row, column 0, or the last column.
- Different objects touch different numbers of border edges (1, 2, or 3; an object cannot touch all 4 edges given the 5x5 size constraint).
- No two objects share a color.

Rule:
- For each object, count how many distinct grid border edges it touches (a number from 1 to 3). Call this count K.
- In the output, arrange each object as K horizontally-adjacent copies of its bounding box pattern, placed in a vertical stack. Objects are ordered top-to-bottom by their K value: objects with K=1 first (at the top), then K=2, then K=3.
- Within the same K group, objects are ordered top-to-bottom by their color's leftmost pixel column position in the input (leftmost first).
- Copies within a row are placed immediately adjacent (no gap). Between each object's row-of-copies, there is exactly one row of black pixels as a separator.
- The output grid width equals 3 times the maximum bounding box width (to accommodate the K=3 case); narrower replicated rows are left-aligned with black fill on the right.

Key insight:
The objects' positions in the input serve a dual purpose: they define the shapes AND encode the replication count through border adjacency. The solver must realize that the number of grid edges touched -- a topological property of placement rather than shape -- determines how many times each object is repeated, and that spatial position in the input is otherwise discarded.
</new_puzzle>

<new_puzzle>
Paint each cell of the output grid with the color of whichever foreground object's convex hull contains that cell; cells inside multiple hulls take the color of the smallest-area hull.

Input structure:
- bg = black background.
- A grid (15x15 to 25x25) contains several sparse foreground objects, each a distinct non-black color. Each object consists of exactly 3 or 4 scattered pixels (not necessarily connected), forming the vertices of a convex polygon (triangle or quadrilateral).
- Objects' vertex pixels may be far apart; their convex hulls may overlap.
- At least 3 objects, at most 6.

Rule:
- For each foreground color, compute the convex hull of its pixels: for 3 pixels this is a triangle, for 4 pixels it is a convex quadrilateral.
- For every cell in the grid, determine which convex hulls contain it (a cell is "inside" a convex hull if it falls within or on the boundary of the polygon).
- If a cell is inside exactly one hull, color it with that hull's color.
- If a cell is inside multiple hulls (overlapping region), color it with the color of the hull having the smallest area.
- If a cell is inside no hull, it remains black.
- The original vertex pixels are part of their hull boundaries and follow the same rule.

Key insight:
The scattered pixels look like random noise, but same-colored pixels are vertices of implicit convex polygons. The solver must mentally connect same-colored dots into triangles/quadrilaterals and realize the output paints their interiors. The overlap resolution rule (smallest area wins) is discovered by examining overlapping regions across examples.
</new_puzzle>

<new_puzzle>
Route each arrow-shaped object in its pointing direction, bouncing off grid edges like a billiard ball, for a number of steps equal to the arrow's body length.

Input structure:
- bg = black background.
- The grid (15x15 to 25x25) contains several arrow-shaped objects, each in a distinct non-black color. Each arrow consists of a straight body segment (horizontal or vertical, at least 2 pixels long and at most 5) plus a one-pixel arrowhead that branches from one end of the body in a perpendicular direction, indicating the arrow's pointing direction.
  - For a horizontal body, the arrowhead is one pixel above or below the last pixel, indicating up or down.
  - For a vertical body, the arrowhead is one pixel left or right of the last pixel, indicating left or right.
- Arrows do not overlap. There are at least 2 and at most 5 arrows.

Rule:
- For each arrow, determine its pointing direction (up, down, left, or right) from the arrowhead position relative to the body.
- Let L = the body length (number of pixels in the body, not counting the arrowhead).
- Starting from the arrowhead pixel, trace a path of L pixels in the pointing direction. If the path reaches a grid boundary before L steps, it reflects (bounces): the component of direction perpendicular to the hit boundary reverses (like a billiard ball), and tracing continues for the remaining steps.
- Draw all L traced pixels in the arrow's color. The original arrow (body + arrowhead) remains unchanged.
- If a traced pixel lands on a cell already occupied by another color, the later-drawn color overwrites. (Processing order: arrows are processed from top-to-bottom, left-to-right by their arrowhead position.)

Key insight:
The arrow encodes both direction and distance: the body length is the trail length, and the arrowhead orientation sets the initial travel direction. The billiard-bounce reflection off grid edges means trails can change direction mid-flight, producing paths that are not simply straight lines -- solvers must simulate the reflection to predict where each trail goes.
</new_puzzle>

<new_puzzle>
Transfer each small shape into the interior cell of the large frame whose border color matches the shape's color, scaling the shape to fit exactly within that cell region.

Input structure:
- bg = black background.
- One large rectangular frame occupies most of the grid. The frame is one pixel thick on each side. Each side of the frame is a different solid color (top edge = one color, bottom edge = another, left edge = another, right edge = another), drawn from: red, blue, green, yellow. The four corners of the frame each show the color of whichever side was drawn last (corners are ambiguous but irrelevant to the rule).
- The frame's interior is divided by single-pixel grey cross-hairs (one horizontal and one vertical grey line) into exactly four equal rectangular cells (2x2 arrangement of cells).
- Outside the frame, scattered on the black background, are exactly four small monochrome shapes, each a different color matching one of the frame's four edge colors. Each shape fits within a bounding box of at most 4x4.

Rule:
- Each shape is placed into the frame cell that is adjacent to the frame edge matching the shape's color:
  - The shape matching the top edge color goes into one of the two top cells.
  - The shape matching the bottom edge color goes into one of the two bottom cells.
  - The shape matching the left edge color goes into one of the two left cells.
  - The shape matching the right edge color goes into one of the two right cells.
- Since each shape matches exactly one edge, and top/bottom narrows to a row (2 cells) while left/right narrows to a column (2 cells), the intersection uniquely assigns each shape to one cell: the top-left cell gets the color shared by top-edge and left-edge matches (but since each shape matches only one edge, the cell is determined by the pair of shapes whose edges intersect at that cell -- specifically, each cell is at the intersection of one horizontal-edge color and one vertical-edge color, and receives the shape whose color matches the horizontal edge).
- Actually, a simpler precise rule: the top-edge-color shape goes into the top-left cell, the right-edge-color shape goes into the top-right cell, the bottom-edge-color shape goes into the bottom-right cell, the left-edge-color shape goes into the bottom-left cell. (Clockwise assignment starting from top to top-left.)
- Each shape is scaled uniformly to exactly fill the target cell dimensions (stretching each pixel of the shape's bounding box to cover multiple pixels), using nearest-neighbor scaling. The shape is painted in its original color; background pixels within the bounding box remain black after scaling.
- The frame, grey dividers, and all areas outside the frame are unchanged. The original shapes outside the frame are erased (replaced with black).

Key insight:
The frame's per-side coloring is the assignment key: each edge color tells which external shape belongs to the adjacent interior region. The clockwise correspondence between edge position and cell position is the hidden rule. The scaling requirement means shapes change size, making it non-trivial to recognize the correspondence purely from shape matching.
</new_puzzle>
