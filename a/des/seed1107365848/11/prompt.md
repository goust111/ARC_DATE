你是一个 ARC-AGI 谜题设计专家。下面的 20 道参考题是难度和机制质量的标杆。仔细阅读每道题的**解题逻辑**（解题者从样例归纳规则的认知过程），而非表面场景。你的新题应在机制深度和隐蔽性上与参考题持平或者实现超越。

每个参考题描述包含：
- `description`：一句话概述题目的变换目标。
- `Input structure`：输入网格中有哪些对象、背景、标记、边界或组件。
- `Rule`：从输入到输出的确定性变换规则。
- `Key insight`：解题所需的核心洞见。

参考题目如下：
<reference_puzzle index="1" source="train" puzzle_id="981add89" lines="10-29">
<description>
Extend each top-row control pixel downward as a vertical toggle ray in its own color.

Input structure:
- The background is the dominant color.
- Non-background pixels on the top row are control pixels, each identifying
  one column and one ray color.
- Foreground regions below may intersect these control columns.

Rule:
- For every top-row control pixel, process the cells below it in the same
  column down to the bottom edge.
- A cell that already has the control color is erased to the background.
- Every other cell on that downward ray is painted with the control color.
- The top-row control pixels themselves remain unchanged.

Key insight:
The top row is a set of vertical ray launchers; each ray toggles only cells
matching its own color and overwrites all other colors along that column.
</description>
</reference_puzzle>

<reference_puzzle index="2" source="eval" puzzle_id="7b5033c1" lines="7-21">
<description>
Straighten a colored snake path into a single-column strip preserving pixel order.

Input structure:
- bg = most common color.
- Non-bg pixels form a single 4-connected path (snake) with exactly two endpoints.
- Each path pixel may be a different non-bg color.

Rule:
- Find the two endpoints (cells with only one path neighbor) of the snake.
- Begin at the top-left-most endpoint and trace the path cell by cell.
- Output an (N×1) column listing each cell's color in traversal order.

Key insight: The path is a simple chain with no branches; choosing the top-left-most endpoint fixes a unique reading direction.
</description>
</reference_puzzle>

<reference_puzzle index="3" source="train" puzzle_id="1d61978c" lines="8-22">
<description>
Recolor the two diagonal foreground-bar orientations, assigning teal to the larger orientation and red to the smaller one.

Input structure
- The grid has one dominant background color and one foreground color.
- Foreground pixels form one-cell-thick diagonal bars.
- Every foreground pixel belongs to exactly one diagonal orientation.

Rule
- Keep the background unchanged.
- Classify each foreground pixel by whether it has a diagonal neighbor on the NW/SE axis or the NE/SW axis.
- Recolor the orientation with more foreground pixels teal and the other orientation red.

Key insight
- The output colors depend on the relative population of the two diagonal orientations, not on a fixed slash direction or position.
</description>
</reference_puzzle>

<reference_puzzle index="4" source="eval" puzzle_id="247ef758" lines="35-46">
<description>
Move colored sprites from the left strip into a framed area at marker-defined targets.

Input structure:
- bg background; rectangular frame on right with colored markers on its edges; sprites in a narrow left strip separated by a constant-color divider column.

Rule:
- For each color, target rows/cols are where both opposite frame edges share that color. Targets are the Cartesian product.
- Erase each sprite from the left and stamp it centered at every target. Higher sprites cover lower ones.

Key insight: One sprite replicates to a grid of row × column marker intersections.
</description>
</reference_puzzle>

<reference_puzzle index="5" source="eval" puzzle_id="faa9f03d" lines="36-52">
<description>
Repair a one-cell-thick colored line network by replacing singleton markers, filling short orthogonal gaps, and redirecting yellow-marked arms to the edge.

Input structure:
- A rectangular grid with a dominant background.
- Line colors form one-cell-thick horizontal/vertical stroke fragments.
- Marker colors are exactly those whose 4-connected components are all singleton pixels.
- Red singleton markers are missing line cells; yellow singleton markers are directional controls next to a line endpoint.

Rule:
- Assign every marker to the nearest visible line color in row/column line-of-sight, preferring an endpoint supported along that same axis.
- For each line color, fill the gap between consecutive facing endpoints in a row or column.
- A yellow control erases the old straight arm on the far side of its endpoint and paints the same color from that endpoint through the control pixel to the grid edge.
- At crossings, yellow-control rays win; otherwise an original-vs-filled crossing uses the more fragmented pre-repair line color, while pure filled crossings use the lower color id.

Key insight: The singleton markers identify which sparse stroke fragments belong together; after they are colored, the intended lines are the axis-aligned gaps between facing endpoints, with yellow controls flipping one arm to the marked side.
</description>
</reference_puzzle>

<reference_puzzle index="6" source="train" puzzle_id="ad173014" lines="97-111">
<description>
Recolor each framed interior mask with the color from the next red frame clockwise around the blue connector cycle.

Input structure
- The grid contains hollow red square frames connected pairwise by blue line components.
- The blue components connect the frames into one cycle, with every frame touching exactly two blue components.
- Each frame interior contains background plus exactly one non-background color mask.

Rule
- Sort the red frames clockwise by their centers around the cycle.
- Keep each frame's interior mask shape in place, but recolor it with the interior color from the next clockwise frame.
- Leave the red frames, blue connectors, background, and interior background holes unchanged.

Key insight
- The colored masks do not move; the blue cycle gives the order in which their colors rotate.
</description>
</reference_puzzle>

<reference_puzzle index="7" source="eval" puzzle_id="a251c730" lines="8-21">
<description>
Stamp source sprites into a destination frame via color-matched singleton markers.

Input structure:
- Two bordered rectangles: source (multi-cell sprites) and destination (singleton markers).
- Each sprite has an anchor pixel — its least-frequent color. Markers share these colors.

Rule:
- Match each destination marker to the source sprite with the same anchor color.
- Translate the sprite so its anchor overlaps the marker position.
- Output is the destination rectangle, interior cleared before stamping.

Key insight: The rarest color per sprite doubles as a template key and positioning anchor.
</description>
</reference_puzzle>

<reference_puzzle index="8" source="train" puzzle_id="522fdd07" lines="8-23">
<description>
Cycle each solid odd-sided square to the next smaller centered square, wrapping a single pixel to a 9x9 square.

Input structure:
- The grid has one dominant background color.
- Every foreground object is a separated 4-connected monochrome solid square.
- Square side lengths are odd and belong to the cycle 1, 3, 5, 7, 9.

Rule:
- Clear the canvas to the background.
- Keep each square's color and center.
- Replace side length n by n - 2 when n > 1; replace side length 1 by side length 9.

Key insight:
- The visible operation is on square size, not color: side lengths cycle 1 -> 9 -> 7 -> 5 -> 3 -> 1.
</description>
</reference_puzzle>

<reference_puzzle index="9" source="eval" puzzle_id="e8686506" lines="172-185">
<description>
Assemble scattered colored fragments into a left-right symmetric rectangle, jigsaw-style.

Input structure:
- bg = most common color; remaining colors form small sprite fragments.
- Optional roles: 4 singleton pixels of one color mark corners; horizontal bars mark top/bottom edges; paired vertical lines mark left/right edges.

Rule:
- Collect all non-bg connected components as jigsaw pieces.
- Determine output rectangle dimensions from total piece area.
- Pack pieces into the rectangle so it is exactly mirror-symmetric (left-right) and same-color pieces never 4-touch.

Key insight: Left-right symmetry plus the no-same-color-touching constraint make the assembly unique; corner/edge markers anchor it when present.
</description>
</reference_puzzle>

<reference_puzzle index="10" source="train" puzzle_id="6d1d5c90" lines="8-23">
<description>
Remove the left control column and roll the remaining image downward by the red marker's row.

Input structure
- The input is a rectangular grid with a one-cell-wide control column on the left.
- That control column is all pink except for exactly one red marker pixel.
- The columns to the right are the image that remains in the output.

Rule
- Delete the left control column.
- Let r be the row index of the red marker in that column.
- Cyclically shift the remaining image downward by r rows.

Key insight
- The red marker's vertical position encodes the row-roll amount; the pink
  column is only a control strip and is removed after reading it.
</description>
</reference_puzzle>

<reference_puzzle index="11" source="train" puzzle_id="ecb67b6d" lines="138-155">
<description>
Recolor the grey diagonal-chain regions in an orange field to teal.

Input structure
The input contains only grey pixels and orange pixels.  Grey pixels form
several 8-connected regions embedded in the orange field.

Rule
Decompose each grey 8-connected region into 4-connected lobes.  Recolor the
whole region teal only when it contains a straight diagonal run of at least
three pixels and those lobes meet only by diagonal corner contacts in a
non-branching diagonal chain or cycle; a single 4-connected region is
recolored only when it is a square diagonal staircase.

Key insight
The teal regions are not selected by size or position; they are the grey
regions whose internal connectivity depends on diagonal corner joins, while
same-edge cracks, rectangles, bars, and branching/noisy regions remain grey.
</description>
</reference_puzzle>

<reference_puzzle index="12" source="eval" puzzle_id="b0039139" lines="29-44">
<description>
Recolor a cropped template and repeat it according to a component-count stencil.

Input structure:
- Solid blue (1) separator rows or columns split the grid into four sections.
- Section 1 is a black-background shape template; section 2 is a component-count stencil.
- Sections 3 and 4 are solid color swatches for output foreground and background.

Rule:
- Crop section 1 to the bbox of its non-black pixels and recolor it with the two swatches.
- Count 4-connected non-black components in section 2.
- Repeat the recolored crop that many times, inserting one background-colored row/column between copies.

Key insight: The second section encodes only the repeat count; the output orientation
is inherited from whether the input separators are columns or rows.
</description>
</reference_puzzle>

<reference_puzzle index="13" source="train" puzzle_id="22425bda" lines="58-76">
<description>
List the colors of overlaid straight lines in their only possible bottom-to-top crossing order.

Input structure
The background is the dominant color. Every foreground color is the visible part
of one full one-cell-thick horizontal, vertical, or 45-degree diagonal line.
Higher lines may hide pixels of lower lines at crossings.

Rule
Recover each full line from its visible collinear pixels. At any cell where
multiple recovered lines cross, the visible color is above every other line
through that cell. Return the unique ordering of all line colors that respects
those above/below constraints and whose consecutive colors are lines that
geometrically cross.

Key insight
The visible crossing colors give a partial layer order, and the intended answer
is the unique Hamiltonian chain through the line-intersection graph from the
bottom layer to the top layer.
</description>
</reference_puzzle>

<reference_puzzle index="14" source="train" puzzle_id="1a244afd" lines="49-68">
<description>
Rotate each magenta source pixel 90 degrees counterclockwise around its matched blue pivot, recoloring the new pixel orange.

Input structure:
- The background is the dominant color.
- The only foreground colors are blue pivot pixels and magenta source pixels.
- Each magenta source can be paired one-to-one with a blue pivot in the same
  row or column.

Rule:
- Remove all magenta source pixels.
- For each matched source/pivot pair, preserve the source distance from the
  pivot but rotate its offset 90 degrees counterclockwise.
- Paint the rotated in-bounds target orange; if the target is outside the
  grid, the source simply disappears.

Key insight:
The blue pixels are rotation centers, and row/column alignment gives a
unique global matching from each magenta source to its pivot.
</description>
</reference_puzzle>

<reference_puzzle index="15" source="train" puzzle_id="4e7e0eb9" lines="31-49">
<description>
Transform each 9x9 panel by mirroring across its center guide line or filling all four block slots from its lone exemplar.

Input structure
The grid is made of 9x9 panels separated by one-cell-wide grid lines. Each
panel has four monochrome 3x3 slots at the quadrant positions, with either a
full non-background center row, a full non-background center column, or no
guide line.

Rule
A panel with a full center column is reflected left-to-right, and a panel
with a full center row is reflected top-to-bottom. A panel with no guide line
has exactly one slot color that appears once and one slot color that appears
three times; recolor all four slots to the once-occurring slot color.

Key insight
The center guide line encodes the mirror axis, while panels without a guide
line are completed by copying the structurally unique slot color.
</description>
</reference_puzzle>

<reference_puzzle index="16" source="train" puzzle_id="dc46ea44" lines="11-25">
<description>
Move the lower sprite and its pink guide into the upper panel, aligning the sprite to the guide's lower-left branch.

Input structure
- The grid has one dominant background color and one full-width separator row.
- Below the separator are exactly two foreground roles: a five-pixel guide marker and one monochrome sprite.
- The guide marker has normalized shape {(0,2), (1,2), (2,0), (2,2), (3,1)} and starts immediately below the separator.

Rule
- Clear the canvas to the background, preserving the separator row.
- Translate the guide marker upward so its top row becomes row 0.
- Move the sprite, preserving its color and shape, so the sprite bbox bottom-right aligns to the translated guide's lower-left branch.

Key insight
- The marker supplies the destination anchor: its lower-left branch cell is where the moved sprite's bbox bottom-right must land.
</description>
</reference_puzzle>

<reference_puzzle index="17" source="eval" puzzle_id="409aa875" lines="68-81">
<description>
Fire projectiles from 3-pixel triangle sprites; mark collisions with maroon or blue.

Input structure:
- bg = most common color. Non-bg 3-pixel sprites shaped as axis-aligned L's or diagonal V's.

Rule:
- Each triangle has an apex and an emission direction determined by its shape: L-shapes fire diagonally away from the missing corner; V-shapes fire along the axis away from the open side.
- A projectile lands exactly 5 cells from the apex in the emission direction.
- If it hits an existing sprite, that entire sprite is recolored to maroon (9).
- If two projectiles land on the same empty cell, that cell becomes blue (1); a lone projectile on empty bg paints maroon (9).

Key insight: The triangle's orientation encodes both direction and target; dual-projectile collisions on open bg produce blue, not maroon.
</description>
</reference_puzzle>

<reference_puzzle index="18" source="train" puzzle_id="48634b99" lines="14-27">
<description>
Advance the marker from one half of a vertical bar to the same half of the next longer bar.

Input structure
- The grid has one dominant background color and exactly two foreground colors.
- One foreground color forms many vertical one-cell-wide bars; the other appears as a single vertical marker segment.
- The marker segment occupies exactly the top or bottom half of its containing vertical bar.
Rule
- Recolor the current marker segment back to the bar color.
- Find the unique all-bar-color vertical bar whose height is two pixels greater than the marked source bar.
- Recolor the same half of that next longer bar to the marker color.
Key insight
- Each bar's total height encodes an ordered size, and the marker advances from a 2k bar to the corresponding 2(k+1) bar while preserving whether it marks the top or bottom half.
</description>
</reference_puzzle>

<reference_puzzle index="19" source="train" puzzle_id="17b866bd" lines="49-67">
<description>
Move each marker from a lattice corner gap into the adjacent enclosed 12-pixel cavity.

Input structure:
- The grid is a repeated teal lattice around black 12-pixel cavities.
- A marker is a non-black pixel placed at the northwest corner gap of one
  complete 5x5 lattice tile; the marker may itself be teal.

Rule:
- For every marked northwest corner gap, clear that marker to black.
- Fill the 12 black pixels in the tile immediately down-right of the marker
  with the marker color.
- Leave the rest of the lattice unchanged.

Key insight:
- The apparent isolated markers are control pixels for the adjacent cavity;
  teal markers are recognized as extra teal pixels occupying a gap that the
  periodic lattice template says should be black.
</description>
</reference_puzzle>

<reference_puzzle index="20" source="train" puzzle_id="e729b7be" lines="8-20">
<description>
Complete the scene by adding the point-reflected copy of the colored object across the central marker.

Input structure
- The grid has odd dimensions, a dominant background, and a center-row marker color appearing only at the left edge, center, and right edge.
- A single divider color fills the center column except for the center marker.
- All remaining foreground pixels form the object to copy, entirely on the left side of the divider.
Rule
- Keep the input unchanged.
- For every object pixel, paint the pixel rotated 180 degrees around the center marker with the same color.
Key insight
- The center marker is the rotation center; the yellow-looking divider is not mirrored, it only separates the source from the empty target side.
</description>
</reference_puzzle>

你的任务：
以参考题为标杆，运用发散性思维进行创新扩展。请发挥创造力，构思全新且巧妙的推理谜题。用中文创造全新的 ARC 谜题描述。
创新方式包括但不限于：
**机制重组**：将多道参考题各自的核心机制提取出来，组合成一个新的多步管道，用到新的场景。
**机制变异**：取几道参考题的核心机制，以机制为基础进行创新并与其他多种机制进行创新重组，使解题路径发生质变而非量变。
**逆向构造**：取一道参考题的输出模式，思考"什么样的输入规则能产生类似的输出结构但需要完全不同的推理过程"。

严格使用下面的 XML 风格格式输出：

<new_puzzle>
- `description`：一句话概述题目的变换目标。
- `Input structure`：输入网格中有哪些对象、背景、标记、边界或组件。
- `Rule`：从输入到输出的确定性变换规则。
- `Key insight`：解题所需的核心洞见。
</new_puzzle>

请一次性生成 5 个 `<new_puzzle>`，若有较多合适的创新，也可以生成更多，不设上限，但不要输出额外解释。

注意事项：
1. 任务不应依赖大量先验知识或复杂计算，而应依赖对模式、结构与空间关系的直觉推理。
2. 关于"计算"的边界：
   - **允许使用**（这些都是人类能心算的视觉量）：数像素 / 孔洞 / 对象 / 邻居（≤10 量级）、比大小、按某个可见属性排序、奇偶判断、取最大 / 最小、用一个计数值作为另一个等量几何量（如 N 个像素 → 长度 N 的射线、面积 N → 边长 N）、判断接触 / 相邻 / 包含、粗略中心位置（簇的视觉中点）、"距离"作为"几步"而不是公式等人类能够直观感受计算的内容。
   - **禁止使用**（这些是算法量而非视觉量，会迫使解题者依赖公式而非视觉直觉）：曼哈顿 / 欧氏 / Chebyshev 距离作为精确查表键、凸包面积、二进制位编码 / 位运算、模运算 / 余数、拓扑距离（BFS / 最短路 / 跳数）、精确几何质心坐标、凸角 / 凹角计数等复杂且不易被看出来的计算都是禁止使用的。
3. 问题必须能从少量示例中推断出来（通常为 3 对训练样例 + 1 个测试样例）。设计时从人类解题者的视觉角度思考：规则涉及的所有关键要素（对象、关系、属性）必须在网格上肉眼可见，不应依赖人眼无法直接观察到的隐含信息（例如颜色id和颜色亮度是禁止使用的）或需要推算的隐藏属性。规则背后的逻辑本身可以有一定挑战性，但解题所需的所有线索必须能从网格中直接获取。
4. 生成的问题描述旨在促进后续用于创建题目网格的代码生成。因此，对输入 / 输出网格的描述以及规则总结必须极其详细；语言风格应易于理解；对于特定对象（如箭头），必须在描述中明确其绘制方式。
5. 描述谜题时避免使用 0 或 1 这类颜色编号；应使用颜色名称，例如 "red"。可用颜色如下：Black、Blue、Red、Green、Yellow、Grey、Magenta、Orange、Azure、Maroon。
6. 5 个新题的核心推理机制应明显不同——即从观察样例到归纳规则的认知路径不同，而不仅仅是场景、对象或颜色不同。
7. 题目难度应来自机制的隐蔽性而非步骤数。用参考题作为难度标尺——如果你生成的题一个样例就能猜到规则，说明机制太透明，需要加深隐蔽性。
8. 优先从参考题的机制出发进行重组或推演，而非从物理隐喻出发。物理隐喻容易导致大量题目归结为一种模式，例如"沿方向移动到阻挡处停止"。
9. 题目的难度优先来自规则的**结构层数**，而不是单一步骤的复杂度。关于"结构层数"的判定标准：**只有解题者需要主动推理才能完成的步骤才算一层**。肉眼一扫即可获得的信息（如"找到缺口在哪一侧"、"数出像素个数"、"识别哪个对象最大"）不构成独立的推理层——它们只是变换的前置输入，不增加认知难度。真正的多层结构是指每一层都需要解题者发现一个非显然的规律。以两层结构为例：
   - ✗ 伪两层："找到缺口方向"→"把东西推到对面" — 第一步肉眼可见，实际只有一层推理。
   - ✓ 真两层："通过孔洞数量匹配颜色图例"→"用匹配到的颜色重绘对象" — 两步都需要非平凡的归纳。

   如果你发现自己写出的题只是"对每个对象，根据它的某个属性做某变换"的单步规则，请引入一个需要主动推理的前置步骤。**用堆叠 tie-break 规则或复杂数学公式来"加难度"是劣解；用多一层需要归纳的结构嵌套来"加难度"才是好解**。
10. 在符合ARC题目和提示词要求的基础上，鼓励尝试创新规则为多层结构的题目，不局限于两层，可以创新三层或四层结构的题目，但不可为了多结构层数违背要求。

生成后对生成的题执行以下四项审查（审查内容不需要保存到文件）：
1. 原创性
  新题不可与上述参考题相同或类似，不可只是替换颜色、方向、对象类型等表层要素。判断标准：新题的核心推理路径应与任何一道参考题不同。
2. 确定性
  规则完全确定，同一输入只能产生唯一输出，每一步操作都有明确定义。
3. 可实现性
  所有必要细节都已给出——对象如何生成、如何识别、如何变换、边界情况如何处理，读完描述后可以直接写代码。
4. ARC 合规性
  网格尺寸合理（≤30×30），使用颜色不超过 10 种，规则复杂度适中，人类能从 3-5 个样例中发现规律。没有使用上述注意事项第二条禁止使用的复杂计算。

