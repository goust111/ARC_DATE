## 第一步：分析与思考

### 基础操作分类

参考题中涉及的基础操作可归纳为以下几类：

1. **空间移动与重力**：对象下落（#5 重力下落、#13 带缓冲下落）、滑动到目标（#16 方块穿越通道）、沿路径移动（#6 沿路径到门、#10 沿轮廓滑行）。
2. **折叠与旋转**：折叠杆件（#7 沿铰链折叠）、旋转放置（#17 按时钟指针旋转）。
3. **投影与延伸**：从标记点向外投影重复图案（#9 周期性延伸）、从矩形长出路径（#20 带端帽的探针）、条纹传播（#14 条纹传播并染色）。
4. **对齐与归一化**：将分散像素对齐到统一基线（#1 双色柱归一化）、上下对齐裁剪（#2 角裁剪选择）。
5. **拼接与组合**：通过连接器对接精灵（#15 连接器对接）、通过不完整掩码的并集恢复完整形状（#18 互补掩码联合）。
6. **标记与标注**：标记地形极值列（#8 最深最浅列标注）、箭头指向的帧端点着色（#11 箭头方向标注帧）、标记颜色转移到邻接区域（#19 标记上色中性区域）。
7. **筛选与计数**：按面积阈值分类对象（#5 大小分类）、按颜色筛选并计数（#12 顶栏筛选与计数条）。
8. **形变与重构**：将垂直像素流重构为金字塔（#4 垂直流→居中金字塔）、交换标记穿越屏障（#3 跨屏障交换标记）。

### 非显而易见的串联方式

- **拓扑特征编码控制参数**：一个对象的孔洞数量、连通分量数或凸缺陷数可以编码另一个操作的步数或方向——拓扑信息通常不是人第一眼注意到的。
- **相对位置差编码变换量**：两个标记之间的偏移量不仅指示方向，还可以作为缩放因子或旋转角度的隐式参数。
- **对称性缺陷作为信息载体**：一个几乎对称的图案中，打破对称的那几个像素恰好编码了操作指令。
- **路径交叉点产生新颜色**：两条不同颜色的路径交叉时，交叉点的颜色可以由混合规则决定，形成一个需要逆向推断路径的谜题。
- **面积比作为选择器**：对象之间的面积比值（而非绝对面积）决定操作的选择分支。
- **边界接触模式编码旋转**：对象触碰网格边界的方式（哪些边、几个像素）可以编码该对象需要的旋转角度。

### 发散性创新方向

- 利用**对象的轮廓形状**（而非面积或颜色）作为隐式匹配键。
- 通过**像素邻域的局部模式**触发全局变换。
- 将**路径长度**映射为输出中的几何参数（如矩形的宽高）。
- 让**缺失像素的位置**（负空间）本身承载关键信息。
- 用**同色对象之间的相对排列顺序**编码操作优先级。

---

## 第二步：生成新谜题

<new_puzzle>
Detect each foreground object's silhouette symmetry axis and reflect the asymmetric pixels to complete the object into its minimal symmetric superset.

Input structure
- The background is the dominant color (e.g. black).
- Several separated 4-connected foreground objects of distinct colors are scattered on the grid.
- Each object is "almost symmetric" about exactly one axis: either a horizontal, vertical, or one of the two diagonal axes. Specifically, when reflected about that axis, at most a few pixels are missing on one side, and no extra pixels exist that would conflict.
- No object has two or more candidate symmetry axes.

Rule
- For each foreground object, determine the unique axis about which it is closest to being symmetric. This is found by checking all four candidate axes (horizontal center, vertical center, main diagonal, anti-diagonal of the bounding box) and selecting the one that requires the fewest added pixels to achieve perfect reflection symmetry.
- Reflect every existing pixel of that object across the chosen axis. If the reflected position is background, paint it with the object's color. If the reflected position is already filled with the same color, do nothing.
- The bounding box may grow to accommodate newly added pixels. Objects never overlap after completion.

Key insight
- The puzzle is not about identifying which pixels to remove but about discovering each object's hidden symmetry axis from the pattern of its "almost symmetric" shape, then filling in the missing mirror pixels. The axis is never marked explicitly — it must be inferred from the shape itself, and different objects may have different axes.
</new_puzzle>

<new_puzzle>
Use each small colored key-shape to unlock the matching lock-region by XOR-stamping the key onto the lock, toggling pixels between foreground and background.

Input structure
- The background is the dominant color (e.g. azure).
- The grid contains several "lock" regions: medium-sized rectangular zones outlined by grey border pixels. Inside each lock region, some cells are filled with a unique non-background, non-grey color (the lock color) and the rest are background.
- Outside all lock regions, there are small free-floating "key" shapes. Each key is a contiguous 4-connected object whose color matches exactly one lock region's interior color.
- Each key shape fits within the bounding box of its matching lock region's interior (excluding the grey border).

Rule
- For each key, find the lock region whose interior color matches the key's color.
- Align the key's bounding box to the top-left corner of the lock region's interior.
- Perform an XOR-stamp: for every cell in the aligned key footprint, toggle the lock interior — if the cell was the lock color, make it background; if the cell was background, make it the lock color.
- Remove the key from its original position (set those cells to background).
- Grey borders and unmatched regions remain unchanged.

Key insight
- The key is not simply stamped additively onto the lock; it toggles pixels. This means the visual result depends on the overlap between the key shape and the existing lock pattern. The solver must understand XOR toggling — where key and lock overlap, pixels disappear rather than reinforce.
</new_puzzle>

<new_puzzle>
Route each colored marker pixel to the nearest same-color port on a central grey scaffold by drawing a deterministic L-shaped path, then erase the original marker.

Input structure
- The background is the dominant color (e.g. black).
- A single large connected grey region (the "scaffold") sits roughly in the center of the grid. Its shape is irregular but simply connected (no holes).
- Along the outer perimeter of the scaffold (the grey pixels that are 4-adjacent to at least one background pixel), several pixels have been replaced with distinct non-grey, non-background colors. These are "ports."
- Scattered in the background far from the scaffold are isolated single-pixel "markers," each matching exactly one port color. Every port color has exactly one matching marker, and vice versa.

Rule
- For each marker, find the unique port pixel on the scaffold perimeter that shares its color.
- Draw a one-pixel-wide path from the marker to the port using the marker's color. The path follows a deterministic L-shaped Manhattan route: first travel along the axis with the larger coordinate difference (vertical if |Δrow| > |Δcol|, horizontal otherwise), then turn once and travel along the remaining axis to reach the port. Ties in coordinate difference are broken by moving horizontally first.
- The path overwrites background pixels only; it never crosses or overwrites the scaffold, other paths, or other markers. (The puzzle guarantees no path conflicts for valid inputs.)
- After drawing, the marker pixel becomes the path's starting end (it stays painted in the marker color).
- The scaffold (including its colored port pixels) remains unchanged.

Key insight
- The challenge is recognizing that the colored pixels on the scaffold edge are destinations (ports), and the isolated pixels in the background are sources (markers). The path-drawing rule uses a deterministic L-shaped routing with a specific axis-priority tie-breaker, which must be inferred from examples where the path bends.
</new_puzzle>

<new_puzzle>
For each enclosed room in a grey partition, find the majority voter color among interior marker pixels, then paint that room's walls in the winning color and erase the voters.

Input structure
- The background is the dominant color (e.g. black).
- A connected network of one-pixel-wide grey lines partitions the grid into several enclosed rectangular "rooms." The grey lines form a rectilinear partition — every line is horizontal or vertical, and rooms are axis-aligned rectangles of varying sizes.
- Inside each room, there are a few scattered single pixels of two or three different non-grey, non-background colors. These are "voter" pixels. Each room contains at least one voter pixel, and every room has a unique strict majority color (no ties).

Rule
- For each room, count the voter pixels of each color present inside it.
- The color with the strictly highest count is the room's "winner."
- Recolor all grey wall pixels forming the boundary of that room to the winning color. Where a wall segment is shared between two adjacent rooms, each wall pixel takes the winner color of the room whose interior it is more naturally associated with: specifically, a wall pixel on a vertical segment takes the winner of the room to its left if there is one, and the room to its right otherwise; a wall pixel on a horizontal segment takes the winner of the room above if there is one, and the room below otherwise. Corner wall pixels (at T-junctions or crossings) take the winner of the room to their top-left.
- After all walls are painted, erase all voter pixels inside rooms (set them to background).

Key insight
- The solver must first identify the room partitions from the grey network, then perform a local majority vote inside each room. The non-obvious step is that shared wall pixels are assigned to a specific room by a consistent directional convention (top/left priority), which must be discovered from example pairs where adjacent rooms have different winners and the shared wall reveals the assignment rule.
</new_puzzle>

<new_puzzle>
Sort colored objects onto vertically stacked shelf rows by ranking their bounding-box heights, placing the tallest on the bottom shelf and the shortest on the top shelf.

Input structure
- The background is the dominant color (e.g. black).
- The grid is divided into two zones by a single horizontal maroon line. The upper zone is the "staging area," and the lower zone contains the "shelf" structure.
- In the lower zone, horizontal red lines span the full grid width at equal vertical intervals, creating N shelf rows of equal height (the spaces between consecutive red lines, and below the last red line to the grid bottom).
- In the staging area, there are exactly N foreground objects of distinct non-background, non-red, non-maroon colors. Each object is a compact connected shape. The N objects have N distinct bounding-box heights (no ties).

Rule
- Rank the objects by their bounding-box height from smallest (rank 1) to largest (rank N).
- Place the rank-k object into the k-th shelf row counting from top to bottom. Within that shelf row, the object is horizontally centered (rounding left on ties). The object's bottom edge rests on the red line at the bottom of that shelf row (or on the grid bottom for the last shelf).
- Clear the staging area (set all staging-area non-maroon pixels to background) after placement.
- Red shelf lines and the maroon divider remain unchanged.

Key insight
- The solver must discover that the assignment of objects to shelves is not based on the objects' horizontal position, color, or area, but strictly on a ranking of their bounding-box heights. The tallest object always ends up on the bottommost shelf, and the shortest on the topmost. This height-to-position sorting rule is the hidden mechanism.
</new_puzzle>

<new_puzzle>
Grow each seed pixel into a filled diamond whose Manhattan radius equals the seed's Manhattan distance to the nearest grid edge.

Input structure
- The background is the dominant color (e.g. black).
- Several isolated single pixels of various non-background colors are scattered across the grid. No two seed pixels share the same color. These are "seed" pixels.
- No seed pixel is located on the grid border (row 0, last row, column 0, or last column); every seed has at least one cell of margin to every edge.

Rule
- For each seed pixel at position (r, c) in a grid of height H and width W, compute its Manhattan distance to the nearest grid edge: d = min(r, c, H-1-r, W-1-c).
- Draw a filled diamond (axis-aligned rotated square) centered at (r, c) with Manhattan radius d. That is, fill every cell (r', c') satisfying |r'-r| + |c'-c| <= d with the seed's color.
- If two diamonds overlap, the seed with the larger radius takes priority (its color overwrites). If radii are equal, the seed closer to the top-left corner (lower row first, then lower column) takes priority.

Key insight
- The size of each diamond is not marked anywhere in the grid — it is implicitly determined by how far the seed sits from the nearest edge. Seeds near the center produce large diamonds; seeds near the edge produce small ones. The solver must discover this distance-to-radius mapping by comparing seed positions and output diamond sizes across examples.
</new_puzzle>

<new_puzzle>
For each pair of same-color pixels on opposite sides of a central divider line, draw a straight bridge between them; if the bridge would cross another bridge, the crossing cell takes the color of the longer bridge.

Input structure
- The background is the dominant color (e.g. black).
- A single vertical grey line spans the full height of the grid, dividing it into a left half and a right half.
- On each side, there are several isolated single-pixel markers. Markers come in distinct non-grey, non-background colors. Each color appears exactly twice: once on the left side and once on the right side.
- No two same-color markers share the same row, and no marker sits on the grey divider.

Rule
- For each color, find the left marker at (r1, c1) and the right marker at (r2, c2).
- Draw a one-pixel-wide path connecting them as follows: from the left marker, move horizontally rightward to the column just left of the grey divider; then move vertically (up or down) to row r2; then move horizontally rightward from just right of the grey divider to the right marker. The path passes through the grey divider column by overwriting that single grey pixel at the turning row with the bridge color.
- If two bridges share a cell (which can only happen on the vertical segments or the divider column), the bridge whose total Manhattan length (|r1-r2| + |c2-c1|) is longer takes color priority at the shared cell.
- After all bridges are drawn, the original marker pixels remain as part of their bridges.

Key insight
- The solver must recognize that same-color pairs on opposite sides are connected by an L-shaped or Z-shaped bridge through the divider, and that crossing conflicts are resolved by total bridge length (longer wins). The divider column is partially consumed by bridge colors, providing a visual clue to the crossing-resolution rule.
</new_puzzle>

<new_puzzle>
Count the number of distinctly-colored neighbors each foreground pixel has and recolor it to the neighbor-count color from a legend strip.

Input structure
- The background is the dominant color (e.g. black).
- The grid has two zones separated by a horizontal grey line. The upper zone is the "legend strip" — a single row containing colored pixels at columns 1 through K (up to 8), each a distinct color, serving as a legend mapping index i to color legend[i].
- The lower zone contains a pattern made from a single foreground color (e.g. blue) on the background. The pattern consists of connected and separated pixel clusters of various shapes.

Rule
- For each foreground pixel in the lower zone, examine its 4 cardinal neighbors (up, down, left, right).
- Count how many of those neighbors are also foreground pixels (not background, not grey). Call this count n (ranging from 0 to 4).
- Recolor that pixel to legend[n]: the color found at column n in the legend strip. (Column 0 has the color for n=0 neighbors, column 1 for n=1, etc.)
- Background pixels and the legend strip remain unchanged.

Key insight
- The solver must discover that each pixel's output color depends on its local connectivity — how many foreground neighbors it has — and that the legend strip at the top is a lookup table mapping this count to a color. Corner pixels (0-1 neighbors), edge pixels (1-2 neighbors), and interior pixels (3-4 neighbors) all receive different colors, creating a visual "heat map" of local density.
</new_puzzle>
