## 第一步：分析与思考

### 参考题中涉及的基础操作

**空间变换类：**
- 平移/滑动（题1、7、11、19）：将对象沿某方向移动到目标位置
- 反射/镜像（题1）：沿边生成镜像副本
- 旋转（题6、7、20）：90度旋转或根据标记确定朝向
- 缩放（题10）：将小图案放大到网格级别

**结构操作类：**
- 拼接/组装（题2）：将分散的碎片拼成完整矩形
- 分割/重组（题4、15）：将对象切分后重新排列
- 堆叠排序（题6）：按某种度量排序后堆叠
- 填充/修复（题5、9、14）：推断缺失区域的内容

**标记-动作映射类：**
- 颜色→目标方向（题19）：颜色决定移动方向
- 位置→模板（题12、17）：角落位置选择固定模板
- 计数→匹配（题20）：数量相等作为配对条件
- 图例→替换（题16）：标记颜色映射到图案

**连接/路径类：**
- 对角连线（题13）：连接同色像素对
- 路径绕障（题18）：绕过障碍连接同色组件
- 射线发射（题13）：从触发点发射垂直射线

**编码/信息隐藏类：**
- 分形自相似（题10）：少数单元的内部图案编码全局布局
- 多集比较（题9）：通过跨区域比较推断缺失值
- 通配符匹配（题2）：某颜色充当通配符角色

### 非显而易见的串联方式

1. **负空间承载语义**：前景对象围成的封闭背景区域本身是一个隐藏的模板。解题者通常关注前景像素，但真正的信息载体可能是被前景包围的"空洞"。这与题5中"灰色是遮挡物而非对象"的思路类似，但将其推广到：负空间的形状本身是可操作的独立实体。

2. **全局统计量作为局部决策依据**：一个像素的变换结果取决于远处像素的统计特性（如全局颜色频率），而非局部邻域。这打破了"变换是局部的"直觉，要求解题者在全局和局部之间建立联系。

3. **几何属性隐式编码数值参数**：对象的拓扑或几何特征（如边数、连通度、对称轴方向）可以隐式编码一个控制后续操作的参数。解题者需要先识别几何属性，再将其映射为操作指令。

4. **并发传播与冲突检测**：多个信号同时在共享介质中传播，冲突点的位置揭示了拓扑距离关系。这与单源填充有本质区别——多源同步传播需要理解"同时性"的概念。

5. **序列编码遍历顺序**：一个颜色序列不是简单的调色板，而是规定了对象访问的顺序，且这个顺序可以循环使用直到所有目标被耗尽。信息从"是什么颜色"转变为"按什么顺序"。

6. **微小附属结构编码变换参数**：一个主体旁边的极小附属物（如单像素臂）的方向编码了主体应执行的变换角度或类型。解题者必须注意到这些容易被忽视的微小标记。

---

## 第二步：全新 ARC 谜题描述

<new_puzzle>
<description>
Extract the shape of the enclosed background cavity inside each colored frame, then stamp that cavity shape at the location of a same-colored lone dot elsewhere on the grid.

Input structure:
- The background is black.
- There are several closed frames made of different foreground colors (e.g., red, blue, green). Each frame is a connected loop of pixels exactly one pixel wide that fully encloses an irregular region of black background pixels. Each frame's bounding box is no larger than 8x8.
- For each frame color, there is exactly one isolated single-pixel dot of the same color located elsewhere on the grid, with no other non-background pixels within a 3-pixel radius of it.

Rule:
- For each colored frame, identify the black pixels fully enclosed by that frame via 4-connected flood fill from the interior, bounded by the frame's color pixels.
- Record these enclosed pixels' positions relative to the top-left corner of the frame's bounding box. This set of relative positions is the "cavity template."
- Locate the lone dot of the same color. Using the dot's position as the top-left anchor of the template's bounding box, draw each pixel of the cavity template in the frame's color onto the grid.
- The original frame and its enclosed region remain unchanged in the output.
- If stamped templates from different colors overlap a cell, the color whose lone dot has the smaller row index (then smaller column index) takes priority.

Key insight:
- The meaningful information is hidden in the negative space: the shape of the black cavity inside each frame, not the frame itself. Solvers must recognize that the enclosed background region is the true template, and the matching-color dot specifies its remote replication site.
</description>
</new_puzzle>

<new_puzzle>
<description>
Fold the grid along a colored divider line, then resolve pixel collisions by keeping whichever color is globally rarer in the original unfolded grid.

Input structure:
- The background is black.
- There is exactly one straight line spanning the full width or full height of the grid, drawn in grey. This is the fold axis. It is either a full horizontal row or a full vertical column.
- On both sides of the grey fold axis, scattered foreground pixels appear in various colors (red, blue, green, yellow, orange, etc.). The two sides are NOT mirror images — they have intentionally different pixel arrangements and color distributions.

Rule:
- Fold the grid along the grey axis line: the side with fewer rows (or columns) folds onto the larger side. If both sides are equal in size, fold the bottom onto the top (for a horizontal axis) or the right onto the left (for a vertical axis).
- The folded side's pixels are mapped by reflection across the axis onto the kept side's coordinate system.
- For each cell in the output grid:
  - If only one side contributes a non-background pixel, keep that color.
  - If both sides contribute different non-background colors, count the total number of pixels of each competing color across the entire original (pre-fold) grid. Keep the color with the strictly smaller total count. If tied, keep the color from the folded (moved) side.
  - If both sides contribute the same color, keep it.
- The grey fold line is removed; the output grid has the dimensions of the kept side.

Key insight:
- The fold axis is visually straightforward, but the collision resolution rule requires a global frequency census of the entire pre-fold grid. Solvers must recognize that rarer colors survive overlaps — a counter-intuitive priority where minority beats majority.
</description>
</new_puzzle>

<new_puzzle>
<description>
Propagate colored signals simultaneously from wire endpoints through a shared grey wire network; where two different signals collide, mark the collision in maroon and halt both.

Input structure:
- The background is black.
- Grey pixels form a network of one-pixel-wide orthogonal wire segments. Wires may branch at T-junctions (3-way) or crossroads (4-way).
- A few wire endpoints (dead-end termini) are colored in distinct non-grey foreground colors (e.g., red, blue, green) instead of grey. Each such colored pixel occupies a position that would otherwise be grey — it is part of the wire network, connected to adjacent grey wire pixels.

Rule:
- From each colored endpoint, propagate a signal along the connected grey wire pixels using simultaneous BFS (all sources advance one step per tick in lockstep).
- Each grey pixel reached by exactly one signal is recolored to that signal's color.
- If two different-colored signals would reach the same grey pixel on the same tick, that pixel becomes maroon, and neither signal propagates further from that point.
- If a signal reaches a pixel already claimed by another signal on a previous tick, the arriving signal stops at the adjacent pixel (it does not overwrite) and that adjacent pixel remains its own color — no maroon is produced, the signal simply halts.
- Grey pixels that are never reached by any signal remain grey in the output. The original colored endpoints retain their colors.

Key insight:
- The grey network is a shared medium with concurrent propagation. The maroon collision points and the boundaries between colored territories reveal relative graph-distances from each source. Solvers must understand that signals travel simultaneously, not sequentially, and that collision timing determines the final coloring.
</description>
</new_puzzle>

<new_puzzle>
<description>
Match each small floating object to the wall niche whose silhouette is its horizontal mirror image, then slide the object into that niche.

Input structure:
- The background is black.
- Along one edge of the grid (e.g., the left column), there is a solid vertical wall drawn in grey, exactly one pixel wide.
- Cut into the grey wall at various vertical positions are rectangular niches: each niche is a small rectangular indentation (2-5 pixels wide, 2-4 pixels tall) carved out of the wall, exposing black background. The inner back wall of each niche is colored in a unique foreground color (a single vertical strip of colored pixels at the niche's deepest column).
- Floating in the open area of the grid are small colored objects, each drawn in a unique foreground color matching exactly one niche's back-wall color. Each object is a compact shape (3-8 pixels, 4-connected) whose bounding box width does not exceed the niche width and whose bounding box height does not exceed the niche height.
- Each object's shape is the horizontal mirror image of the niche's empty silhouette (the black cavity shape when viewed from the open side). Specifically, if you reflect the object left-to-right, it exactly fills the niche's empty space.

Rule:
- For each floating object, find the niche whose back-wall color matches the object's color. The object's horizontally mirrored shape will exactly fill that niche's cavity.
- Horizontally mirror the object, then translate it so that it fills the niche cavity precisely, with the mirrored object's pixels occupying the black pixels of the niche.
- Clear the object's original floating position.
- The grey wall and colored back-wall pixels remain unchanged.

Key insight:
- Color matching alone identifies which niche belongs to which object, but the spatial fit requires recognizing that each object is the mirror image of its niche's cavity. Solvers must notice that the objects are not just "placed into" niches — they must be reflected first to fit, and this reflection is what makes the correspondence non-obvious when comparing shapes directly.
</description>
</new_puzzle>

<new_puzzle>
<description>
Use each colored region's area to determine how many pixels to erode inward from its border, leaving only the interior core; then recolor the core using the color of the region's smallest orthogonal neighbor.

Input structure:
- The background is black.
- The grid is partitioned into 4-6 large irregular regions (each 15-50 pixels), tiling the grid with no gaps or overlaps. Each region is a distinct foreground color (e.g., red, blue, green, yellow, orange). Regions share borders with their orthogonal neighbors.
- No black background pixels exist in the input — the entire grid is covered by colored regions.

Rule:
- For each region, compute its area A (number of pixels).
- Compute the erosion depth E = floor(A / 10). If E is 0, set E = 1.
- Erode the region inward by E pixels: remove all pixels that are within Manhattan distance E of any pixel belonging to a different region. The removed pixels become black.
- The surviving interior pixels form the "core." If the core is empty (the region is too small or thin to survive erosion), leave the entire region as black.
- Recolor each surviving core: find all regions that share a border with the original (pre-erosion) region. Among these neighbors, pick the one with the smallest area. Recolor the core to that neighbor's original color.
- If there is a tie for smallest neighbor, pick the neighbor whose color appears first in the order: blue, red, green, yellow, orange, magenta, azure, maroon.

Key insight:
- The area of each region simultaneously controls two things: how much of it survives (via erosion depth) and indirectly which neighbor it inherits its color from (since neighbor sizes determine the recoloring). Larger regions erode more deeply but survive; smaller regions erode less but may vanish if they are thin. The recoloring rule then cross-links regions by their neighborhood relationships — the output color of a core comes from a different region entirely.
</description>
</new_puzzle>

<new_puzzle>
<description>
Read the directional arrow implied by each L-shaped triomino to determine a ray-casting direction, then extend a ray from the triomino's corner cell until it hits another object, coloring the ray in the triomino's color.

Input structure:
- The background is black.
- Several L-shaped triominoes (each exactly 3 pixels occupying 3 cells of a 2x2 box) are scattered on the grid, each in a distinct foreground color.
- An L-shaped triomino has exactly one "corner cell" — the pixel that has two orthogonal neighbors within the triomino. The missing cell of the 2x2 box (the empty fourth cell) defines the arrow direction: the triomino "points" from its corner cell toward the missing cell.
- There are also several small rectangular blocks (2x2 or 2x3 solid rectangles) in grey scattered across the grid. These serve as ray targets/obstacles.

Rule:
- For each L-shaped triomino, identify its corner cell and the direction toward the missing cell of its 2x2 bounding box. This is the "arrow direction" (one of four cardinal directions: up, down, left, right).
- From the corner cell, cast a ray in the arrow direction. Draw each ray pixel in the triomino's color, continuing until the ray reaches either a grey obstacle pixel or the grid boundary.
- The ray stops at the cell immediately before the grey obstacle (it does not overwrite the grey). If the ray reaches the boundary, it stops at the last valid cell.
- The triomino pixels themselves remain unchanged. Rays from different triominoes that cross the same cell: the ray drawn later (triominoes processed top-to-bottom, left-to-right by corner cell position) overwrites the earlier one.

Key insight:
- Each L-triomino is a disguised directional arrow. The "missing corner" of the 2x2 bounding box is the key to determining the ray direction — solvers must recognize that the empty cell (negative space of the bounding box) encodes directionality. This is subtle because all four orientations of the L-shape look similar at first glance, and the directional information is in what is absent, not what is present.
</description>
</new_puzzle>

<new_puzzle>
<description>
Identify which row or column of a grid-of-cells contains the unique pattern, then replicate that pattern to replace all other rows or columns.

Input structure:
- The background is black.
- The grid is divided into an NxN arrangement of equal-sized rectangular cells by single-pixel-wide grey separator lines (both horizontal and vertical), forming a regular tiling (N is typically 3-5).
- Each cell contains a small pattern made of one or two foreground colors against a black interior background.
- All cells in the same row (or same column) of the cell-grid share an identical interior pattern. Different rows (or columns) generally have different patterns.
- Exactly one row (or column) of cells has a pattern that appears only once — it is unique. All other rows (or columns) share a single repeated pattern.

Rule:
- Determine whether the cell-grid is organized by rows or by columns (i.e., whether same-row cells match or same-column cells match).
- Find the row (or column) of cells whose shared pattern is unique — it appears in exactly one row (or column), while all other rows (or columns) use a different common pattern.
- Replace every cell in the grid with the unique row's (or column's) pattern, keeping the grey separator lines unchanged.

Key insight:
- The grid's cell arrangement suggests comparing cells within rows versus within columns to determine the grouping axis. Once the axis is found, the solver must identify the "odd one out" row/column — the one whose pattern is different from all others. The transformation broadcasts this minority pattern everywhere, which is counterintuitive because the minority replaces the majority.
</description>
</new_puzzle>
