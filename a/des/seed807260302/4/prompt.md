你是一个 ARC-AGI 谜题设计专家。下面的 20 道参考题是难度和机制质量的标杆。仔细阅读每道题的**解题逻辑**（解题者从样例归纳规则的认知过程），而非表面场景。你的新题应在机制深度和隐蔽性上与参考题持平或者实现超越。

每个参考题描述包含：
- `description`：一句话概述题目的变换目标。
- `Input structure`：输入网格中有哪些对象、背景、标记、边界或组件。
- `Rule`：从输入到输出的确定性变换规则。
- `Key insight`：解题所需的核心洞见。

参考题目如下：
<reference_puzzle index="1" source="eval" puzzle_id="c4d067a0" lines="9-23">
<description>
Complete a grid of colored blocks to match a minimap legend in the top-left corner.

Input structure:
- bg = most common color.
- Top-left corner: a small sprite of single colored pixels forming a minimap pattern on a regular lattice.
- Main area: uniformly colored square blocks (NxN) arranged on a regular grid; some positions are empty (bg).

Rule:
- The legend pixels define a color pattern; each pixel's lattice position maps to a block position in the main grid via a fixed scale factor and offset.
- Existing colored blocks confirm the scale/offset alignment.
- For every legend pixel whose corresponding grid position is empty, place a new block of that color.

Key insight: The top-left pixel pattern is a scaled-down minimap of the block layout; the mapping is inferred from blocks already present.
</description>
</reference_puzzle>

<reference_puzzle index="2" source="train" puzzle_id="f3e14006" lines="21-39">
<description>
Build the woven rectangle determined by the two marked cells on each crossed bar.

Input structure
A black-background grid contains one long horizontal bar and one long vertical bar.
Each bar has a dominant base color and exactly two same-colored marker pixels.

Rule
The horizontal marker pair gives the output column span.  From the horizontal bar
row, walk toward the farther vertical marker; rows and columns alternate every
other cell.  Even step/even offset cells use the horizontal marker color until
the nearer vertical marker is reached, then use the vertical marker color.
Odd row steps use the horizontal base on marker columns and the original
crossing color on spacer columns.  Spacer columns on even row steps use the
vertical base, except at the switch row where they use the crossing color.

Key insight
The original cross cell records which base color wins at base/base lattice
crossings; the nearer vertical marker is the color-switch boundary.
</description>
</reference_puzzle>

<reference_puzzle index="3" source="train" puzzle_id="66ac4c3b" lines="57-72">
<description>
Complete the sparse-color reflected copy of the foreground motif across its matching marker row or column.

Input structure
- The grid has one background color and two foreground colors.
- One foreground color appears only as a sparse marker row or column.
- The other foreground color contains the same sparse marker positions on a parallel row or column, plus the motif on the far side of the sparse markers.

Rule
- Keep the original grid unchanged.
- Reflect every motif pixel of the larger foreground color across the two parallel marker lines.
- Paint the reflected motif pixels with the sparse marker color.

Key insight
- The two equal marker rows or columns are paired landmarks: the sparse-color marker line borders the source motif, and the matching source-color marker line anchors the reflected copy.
</description>
</reference_puzzle>

<reference_puzzle index="4" source="train" puzzle_id="46c35fc7" lines="21-34">
<description>
Rotate the colors around every 3x3 foreground ring, with corners and side-centers cycling in opposite directions.

Input structure
- The grid has a dominant background color.
- Each 4-connected foreground object is a complete 3x3 ring: eight foreground pixels around a background center.

Rule
- For each ring, move corner colors one step around the four corners.
- Move side-center colors one step around the four side centers in the opposite direction.
- Leave the background center and all other background pixels unchanged.

Key insight
- Corners and side centers form two separate four-cycles on the same 3x3 ring.
</description>
</reference_puzzle>

<reference_puzzle index="5" source="train" puzzle_id="ad3b40cf" lines="72-87">
<description>
Mirror the smallest non-axis foreground color across the full straight axis.

Input structure
- The background is the dominant color.
- One foreground color forms exactly one full row, column, main diagonal, or anti-diagonal reflection axis.
- The remaining foreground colors include a unique least-populated color to copy.

Rule
- Preserve the input grid.
- Select every pixel of the unique smallest non-axis foreground color.
- Reflect those pixels across the full straight axis and paint the reflected positions with the same color.

Key insight
- The axis color marks the mirror line, while the color population chooses which set of objects is duplicated.
</description>
</reference_puzzle>

<reference_puzzle index="6" source="train" puzzle_id="f1bcbc2c" lines="40-55">
<description>
Paint the top-connected middle channel between two orange path components teal.

Input structure
- The background is black.
- One foreground color forms exactly two 4-connected one-pixel-wide path components.
- Optional non-background control pixels may sit in the channel between the two paths.

Rule
- Dilate each path component by one cell in the 8-neighborhood and intersect the two dilations; this gives the one-cell-wide middle channel.
- A control pixel in that channel is a blocker only when its 4-neighbors touch fewer than both path components; otherwise it is part of the channel and is repainted.
- After removing blockers, paint the channel component connected to the top edge teal and keep all other pixels unchanged.

Key insight
- The desired pixels are not found by region filling; they are the geometric overlap of the two paths' one-cell halos, with weakly attached control pixels splitting the channel.
</description>
</reference_puzzle>

<reference_puzzle index="7" source="train" puzzle_id="e3f79277" lines="8-23">
<description>
Replace a corner L with a same-color right-triangle outline in the corresponding corner of an enlarged canvas.

Input structure:
- The input is a square grid with one background color and one foreground color.
- The foreground is a one-pixel-thick L inside a square bbox.
- The L touches exactly one horizontal grid edge and one vertical grid edge, identifying a corner.

Rule:
- Fill a square output canvas of side 3n-2, where n is the input side length, with the background color.
- Double the L leg length to get the triangle side length.
- Draw the two perpendicular triangle legs on the same output corner edges and connect their free endpoints with the diagonal.

Key insight:
- The input L encodes the corner, color, and side length of the triangle outline; the missing side is the diagonal between the L's two free endpoints.
</description>
</reference_puzzle>

<reference_puzzle index="8" source="train" puzzle_id="bae5c565" lines="10-28">
<description>
Erase the top palette row and redraw it below as a widening wedge centered on its missing cell.

Input structure
- The top row is a palette row with exactly one background gap.
- Below it, the only foreground is one vertical marker bar of a single color.
- The marker bar is either in the gap column or immediately beside it.

Rule
- Clear the grid to background.
- Starting at the top of the marker bar, each next row copies one more
  column of the top-row palette on both sides of the gap.
- Fill the gap column with the marker color, and keep the vertical marker bar.

Key insight
- The lone background cell in the top row is the center of the expanding
  copy; the lower vertical bar provides the center color and the row where
  the expansion begins.
</description>
</reference_puzzle>

<reference_puzzle index="9" source="eval" puzzle_id="bf45cf4b" lines="7-19">
<description>
Scale up a monochrome control sprite by stamping a polychrome tile at each filled pixel.

Input structure:
- bg = background color.
- Two foreground objects (8-connected): a monochrome rectangular sprite with bg holes (control), and a polychrome square tile (block).

Rule:
- For each cell in the control sprite's bbox: if filled, stamp a copy of the block tile; if bg, leave a tile-sized bg region.
- Output grid dimensions are (control_h × tile_h) by (control_w × tile_w).

Key insight: The control sprite acts as a binary mask that gets "pixel-scaled" by the tile — each pixel expands to a full tile copy or empty space.
</description>
</reference_puzzle>

<reference_puzzle index="10" source="train" puzzle_id="5d588b4d" lines="10-28">
<description>
Expand the top-left horizontal bar into a wrapped sequence of increasing and decreasing foreground runs.

Input structure:
- The grid has a background color and exactly one foreground color.
- All foreground pixels form one contiguous horizontal bar in the top row.
- The bar starts at the left edge; its length is the control value n.

Rule:
- Build a one-dimensional sequence of foreground runs with lengths
  1, 2, ..., n, n-1, ..., 2, 1.
- Separate adjacent runs by one background pixel.
- Wrap the sequence row-major across the input width and pad the final row
  with background pixels.

Key insight:
The output preserves the input width; the bar length controls the run-length
diamond, and the apparent 2D pattern is just that 1D sequence wrapped into rows.
</description>
</reference_puzzle>

<reference_puzzle index="11" source="eval" puzzle_id="981571dc" lines="6-19">
<description>
Reconstruct a 4-way symmetric mosaic by filling missing (black) pixels.

Input structure:
- A colored grid exhibiting bilateral symmetry along both a vertical and a horizontal axis.
- Some pixels are erased to color 0 (black), creating gaps in the pattern.

Rule:
- Infer the vertical and horizontal reflection axes from the visible (non-zero) pixels.
- For each black pixel, recover its color from the mirrored position across either axis or both.
- When multiple mirrors yield candidates, use the majority color.

Key insight: The reflection axes are not given — they must be deduced from the surviving pixels by finding the axes that best preserve the existing symmetry.
</description>
</reference_puzzle>

<reference_puzzle index="12" source="train" puzzle_id="af726779" lines="17-35">
<description>
Grow an alternating pink/orange triangular trace downward from the horizontal orange seed row.

Input structure:
- The grid has a green background and one horizontal row containing all orange foreground pixels.
- The seed row may contain isolated orange pixels and adjacent orange runs.
- Pink is absent from the input and is used for the first generated row.

Rule:
- Copy the input grid.
- Two rows below the current active row, place a child pixel between each active pair
  whose middle cell in the current row is background.
- Alternate generated row colors between pink and orange, stopping when no children
  can be generated or the next row would leave the grid.

Key insight:
- Only an active-background-active triplet in the current row reproduces; contiguous
  active runs do not produce children through occupied middle cells.
</description>
</reference_puzzle>

<reference_puzzle index="13" source="train" puzzle_id="2601afb7" lines="30-45">
<description>
Rotate each vertical bar's color one slot right and its height one slot left.

Input structure
- A dominant background surrounds several monochrome one-cell-wide vertical bars.
- Every bar is bottom-aligned and occupies a distinct column.
- The ordered bar slots are read from left to right.

Rule
- Keep the grid shape and the bar columns unchanged.
- At slot i, use the color from the previous slot cyclically.
- At slot i, use the height from the next slot cyclically.

Key insight
- Color and height are independent cyclic attributes of the ordered bar slots:
  colors rotate right while heights rotate left.
</description>
</reference_puzzle>

<reference_puzzle index="14" source="eval" puzzle_id="dbff022c" lines="275-288">
<description>
Fill hollow sprites' interiors using a color-palette legend on the grid edge.

Input structure:
- bg = most common color. Several hollow sprites with distinct border colors.
- A small 2×N or N×2 palette block sits on one grid edge; one row/column lists border colors, the other lists corresponding fill colors.

Rule:
- Locate the palette legend on a grid edge and read it as a border→fill color mapping.
- Find enclosed bg regions (holes) inside sprites that don't touch the grid boundary.
- For each hole, identify the most common adjacent border color, look it up in the palette, and fill the hole with the mapped fill color.

Key insight: The palette's edge position and orientation determine reading direction; border colors in the palette uniquely key into fill colors.
</description>
</reference_puzzle>

<reference_puzzle index="15" source="eval" puzzle_id="291dc1e1" lines="7-22">
<description>
Reorder and stack rectangles by proximity to two colored edge lines.

Input structure:
- bg = most common color.
- Two edge lines (h + v) meet at a corner marked by a unique color.
- "Blue" line is color 1 if present, else the longer line; other is "red."
- Solid rectangles in the interior.

Rule:
- Sort rectangles by (distance to blue edge, distance to red edge).
- If the blue edge is vertical, rotate each rectangle 90° CCW.
- Stack top-to-bottom, centered horizontally.

Key insight: Edge-line colors encode a coordinate system; proximity to each axis determines stacking order and rotation.
</description>
</reference_puzzle>

<reference_puzzle index="16" source="train" puzzle_id="54dc2872" lines="52-69">
<description>
Move each marked L-shaped sprite onto the separate pixel matching its marker color.

Input structure
- The background is the dominant color.
- Foreground has 4-connected multicolor sprites and isolated one-pixel controls.
- Each sprite has one repeated body color in an L shape plus one embedded marker
  pixel of a different color.

Rule
- If the embedded marker color also appears as a separate isolated control pixel,
  translate the whole sprite so the body L-bend lands on that control pixel.
- Clear each moved sprite from its old position, clear its consumed control pixel,
  then paste all translated sprites; sprites without a matching control stay put.

Key insight
- The marker color pairs a sprite with its destination control, while the L-bend
  gives the orientation-independent anchor cell to place on that destination.
</description>
</reference_puzzle>

<reference_puzzle index="17" source="train" puzzle_id="996ec1f3" lines="16-31">
<description>
Compress the noisy four-region board split by a solid cross into a 3x3 color map.

Input structure:
- A single full row and a single full column have the same solid separator color.
- The separator cross divides the input into four nonempty rectangular regions.
- Each region has a unique dominant color, with other colors acting as noise.

Rule:
- Find the unique solid same-color row/column pair.
- Fill the middle row and middle column of the 3x3 output with the separator color.
- Put the unique dominant color of each quadrant into the corresponding corner.

Key insight:
- The input grid is a noisy enlarged version of the desired 3x3 layout; the cross marks the coarse cell boundaries.
</description>
</reference_puzzle>

<reference_puzzle index="18" source="eval" puzzle_id="898e7135" lines="7-20">
<description>
Inflate a frame's hole-layout and fill each hole with its matching piece.

Input structure:
- bg = most common color. Largest component is a monochrome frame enclosing bg holes.
- Outside pieces composed of k×k blocks; k = side of smallest solid square piece.

Rule:
- Extract connected hole regions inside the frame as slots.
- Match each piece to a slot by comparing block-level shapes under rotation/flip.
- Output the layout scaled by k, with pieces placed in slots; frame color as background.

Key insight: Pieces are k×k-block-scaled versions of holes; downsampling reveals shape matches under all 8 orientations.
</description>
</reference_puzzle>

<reference_puzzle index="19" source="train" puzzle_id="4df5b0ae" lines="27-41">
<description>
Pack the colored sprites along the bottom edge from smallest to largest.

Input structure
The grid has a dominant background color and several monochrome 4-connected
foreground sprites; a full-grid one-cell frame may surround the scene.

Rule
Remove any full-grid rectangular frame, sort the remaining sprites by pixel
area from smallest to largest, and paste their bounding-box sprites left to
right with each sprite bottom-aligned.

Key insight
The original locations do not determine the final slots; each sprite's area
gives the packing order, while its shape and color are preserved.
</description>
</reference_puzzle>

<reference_puzzle index="20" source="train" puzzle_id="f0100645" lines="51-68">
<description>
Slide each inner colored object horizontally into the same-colored vertical wall until it touches an obstacle.

Input structure
- The background is the dominant color.
- The only foreground colors are two non-background colors.
- One foreground color fills the left column and the other fills the right column.
- Extra 8-connected objects of each wall color appear away from that color's wall.

Rule
- Clear the extra objects from their original positions.
- For each wall color, process its 8-connected extra objects from nearest to farthest from the matching wall.
- Slide each object horizontally toward its matching wall as far as possible without overlapping the wall or an already settled object of the same color.
- Keep the two vertical wall columns fixed.

Key insight
- The side columns act like color-specific gravity walls: same-colored objects compact sideways against their own wall while preserving each object's shape.
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

