你是一个 ARC-AGI 谜题设计专家。下面的 20 道参考题是难度和机制质量的标杆。仔细阅读每道题的**解题逻辑**（解题者从样例归纳规则的认知过程），而非表面场景。你的新题应在机制深度和隐蔽性上与参考题持平或者实现超越。

每个参考题描述包含：
- `description`：一句话概述题目的变换目标。
- `Input structure`：输入网格中有哪些对象、背景、标记、边界或组件。
- `Rule`：从输入到输出的确定性变换规则。
- `Key insight`：解题所需的核心洞见。

参考题目如下：
<reference_puzzle index="1" source="train" puzzle_id="e729b7be" lines="8-20">
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

<reference_puzzle index="2" source="train" puzzle_id="da6e95e5" lines="70-81">
<description>
Extract the central object from the marker cluster whose markers have the same color.

Input structure
The grid has a dominant background and several clusters; each cluster has four
same-color corner marker objects around one central monochromatic object.
Rule
Find every central object enclosed by four aligned corner markers of one color,
then choose the unique one whose object color equals the marker color.
Key insight
The answer is not the largest or most frequent object; it is the center of the
self-colored cluster, cropped to that object's bounding box with background kept.
</description>
</reference_puzzle>

<reference_puzzle index="3" source="eval" puzzle_id="a32d8b75" lines="262-275">
<description>
Assemble and place enlarged sprites on a canvas using legend-panel instructions.

Input structure:
- Color-6 separators divide the grid into a left legend column and a main canvas.
- Each legend panel has (top to bottom): a unit sprite, a frame template, a 5x5 bordered square with a position-marker pixel, and a 3x3 rotation clock-hand.

Rule:
- Tile the unit sprite into each active cell of the frame to build an enlarged composite.
- Rotate the composite per the clock-hand direction (up=0, right=90, down=180, left=270 degrees).
- Place it on the canvas anchored at the position encoded by the marker pixel's row/col in the bordered square's 3x3 inner grid.

Key insight: The marker pixel's position in a 3x3 grid jointly encodes vertical and horizontal anchor alignment (start/middle/end).
</description>
</reference_puzzle>

<reference_puzzle index="4" source="train" puzzle_id="2f767503" lines="66-81">
<description>
Erase each target-color object crossed by the ray pointed by the bar-and-control marker.

Input structure:
- The background is the dominant color.
- There is one three-pixel straight marker bar and one singleton control pixel adjacent to the bar center.
- The remaining non-background color forms multiple 4-connected target objects.

Rule:
- Cast a straight ray from the marker bar center in the direction away from the adjacent control pixel.
- Find every 4-connected target-color object that intersects that ray.
- Recolor all pixels of those hit target objects to the background; all other pixels stay unchanged.

Key insight:
- The whole object is erased when the center ray touches any of its pixels; the full marker bar width is not swept.
</description>
</reference_puzzle>

<reference_puzzle index="5" source="train" puzzle_id="3bd292e8" lines="13-29">
<description>
Recolor wall-separated background regions alternately, starting with grey in the lower-left region.

Input structure
- The grid has one dominant background color and one foreground wall color.
- The wall pixels form one or more 4-connected one-cell-thick segments.
- Each wall segment separates exactly two 4-connected background regions.
Rule
- Keep all wall pixels unchanged.
- Label the 4-connected background regions and connect two labels when they
  touch opposite sides of the same wall segment.
- Color the region containing the lower-left corner grey, then flip between
  grey and green across every wall adjacency.
Key insight
- The red walls define a bipartite map of regions; the lower-left background
  region fixes which side of the alternating coloring is grey.
</description>
</reference_puzzle>

<reference_puzzle index="6" source="eval" puzzle_id="16de56c4" lines="9-34">
<description>
Fill periodic progressions that shoot from a unique grid edge.

Input structure:
- bg = most common color.  Sparse non-bg dots of several colors.
- All repeating colors originate from the same grid edge (top, bottom,
  left, or right — the one with the most non-bg dots).
- Each repeating color has exactly 2 dots per shooting-direction line;
  their gap defines the step.
- Every non-bg dot lies on a shooting-direction line that contains a
  repeating color.

Rule:
- Determine the unique shooting edge; left/right → rows, top/bottom → columns.
- For each shooting-direction line, the repeating color has exactly 2 dots;
  step = gap between them, offset = first position mod step.
- A different-colored dot aligned to the same beat (pos % step == offset) is
  a companion: it absorbs the ray and becomes the fill color, limiting the
  fill range to the span between repeating and companion dots.
- An unaligned dot is a bystander and left untouched.
- Without a companion the repeating color fills the entire line.

Key insight: the shooting edge is identified by counting boundary dots;
the gap between the two marker dots is the period; a companion sharing the
modular class acts as a wall that stops the ray and dictates the paint color.
</description>
</reference_puzzle>

<reference_puzzle index="7" source="eval" puzzle_id="fc7cae8d" lines="9-21">
<description>
Align a central rectangle to two colored frame-edge controls via rotation/reflection.

Input structure:
- bg grid with two solid-colored border edges (distinct non-bg colors).
- A central filled rectangle with 1px colored guide edges on two sides, colors matching the frame controls.

Rule:
- Rotate/mirror the object so its guide edges face the same sides as the matching frame controls.
- Crop off the 1px guide edges from aligned sides.

Key insight: Guide-edge colors encode target frame sides; the correct dihedral transform matches both simultaneously.
</description>
</reference_puzzle>

<reference_puzzle index="8" source="eval" puzzle_id="6ffbe589" lines="83-98">
<description>
Crop a board cluster and rotate its color layers based on outside cue sprites.

Input structure:
- bg = most common color.
- A main multi-color board cluster (largest foreground region).
- Small monochrome cue sprites outside the board for the colors that rotate.

Rule:
- Tight-crop around the board cluster.
- For each color layer inside the crop, find its same-color cue outside.
- Rotate that layer clockwise by (cue pixel count mod 4) × 90°.
- No outside cue → no rotation.

Key insight: Cue sprite size mod 4 encodes rotation amount for its matching color layer.
</description>
</reference_puzzle>

<reference_puzzle index="9" source="train" puzzle_id="18447a8d" lines="33-48">
<description>
Move each right-side colored shape into the left slot whose template color is its exact complement.

Input structure:
- The grid has two equal-width slots separated by one background column.
- Non-background rows form horizontal bands separated by all-background rows.
- Each band has a shared template color in the left slot and one colored source shape in the right slot.

Rule:
- For each band, define its destination shape as the cells in the left slot not occupied by the template color.
- Normalize every right-side source shape to right-slot coordinates and match it to the unique equal destination shape.
- Clear the original right-side source pixels and paint the matched left-slot complement with the source color.

Key insight:
The right-side objects are not placed by row order; each one completes the unique left template whose missing cells have the same shape.
</description>
</reference_puzzle>

<reference_puzzle index="10" source="train" puzzle_id="522fdd07" lines="8-23">
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

<reference_puzzle index="11" source="eval" puzzle_id="45a5af55" lines="5-18">
<description>
Build nested square frames from a stack of horizontal color stripes.

Input structure:
- Full-width horizontal stripes of distinct non-bg colors, stacked vertically.
- Each stripe has uniform color and variable height (1-4 rows).

Rule:
- Read stripes bottom to top.
- Bottom stripe becomes the center solid square with side length = stripe height.
- Each subsequent stripe wraps the current shape with a square frame of thickness = stripe height, in that stripe's color.

Key insight: Stripe height doubles as both frame thickness and (for the innermost) center-square side length; output dimensions are fully determined by the stripe heights.
</description>
</reference_puzzle>

<reference_puzzle index="12" source="train" puzzle_id="22806e14" lines="23-37">
<description>
Mark the center of each odd-sized solid square with the plus marker color, erasing the marker only when it is outside the marked-center field.

Input structure:
- The dominant color is background.
- Exactly two foreground colors are present: one unique 3x3 plus marker and one color forming separated solid square objects.

Rule:
- For every non-marker 4-connected object, assert it is a solid square.
- If the square side length is odd, recolor its center pixel to the plus marker color; even-sided squares stay unchanged.
- Preserve the plus marker when its center lies inside the bounding box of the odd-square centers; otherwise erase it to background.

Key insight:
- Odd-sided solid squares have a single visible center pixel, and the plus supplies the color used to identify those centers.
</description>
</reference_puzzle>

<reference_puzzle index="13" source="train" puzzle_id="e4888269" lines="67-88">
<description>
Recolor every target pixel by following the ordered two-column substitution legend.

Input structure
- A dominant background surrounds one multicolor vertical legend: a connected,
  fully filled two-column block with one color pair per row.
- Other non-background pixels or objects are the targets; divider or frame
  pixels are ordinary target-side structure that remain unchanged unless their
  color is rewritten by the legend.

Rule
- Read the legend rows from top to bottom as substitutions from the left
  column color to the right column color.
- For each non-background pixel outside the legend, scan those rows in order;
  whenever the current color equals the row's left color, replace it with the
  row's right color and keep scanning lower rows.
- Leave the legend itself and background pixels unchanged.

Key insight
- The same color may be rewritten multiple times, but only by substitutions
  that appear later in the vertical legend, so the row order is part of the
  symbol instruction.
</description>
</reference_puzzle>

<reference_puzzle index="14" source="eval" puzzle_id="c4d067a0" lines="9-23">
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

<reference_puzzle index="15" source="eval" puzzle_id="db0c5428" lines="32-51">
<description>
Expand a 3x3 block motif into a 5x5 block motif around its missing center.

Input structure:
- Teal background surrounds one 9x9 foreground motif.
- The motif is a 3x3 array of 3x3 blocks; the middle block is empty teal.
- Colored pixels immediately around the empty middle block encode the border
  of the missing center block.

Rule:
- Keep the original 9x9 motif in place.
- Copy each of the eight surrounding 3x3 blocks to the opposite outer slot
  of the implied 5x5 block grid, without rotating it.
- Fill the center block from the eight pixels around the hole; its center is
  the motif's majority foreground color.

Key insight: The visible 9x9 motif is the middle of a larger periodic
block-level pattern, and the empty center is reconstructed from its immediate
inner boundary.
</description>
</reference_puzzle>

<reference_puzzle index="16" source="train" puzzle_id="758abdf0" lines="54-69">
<description>
Advance short marker bars away from the full edge bar, turning completed two-pixel markers into edge-color marks at the opposite side.

Input structure
- The background is the dominant color.
- One non-background color forms a complete one-cell-thick bar on exactly one outer edge.
- The other foreground color appears only in the first two cells inward from that bar, lane by lane.

Rule
- In each lane perpendicular to the edge bar, a one-pixel marker adjacent to the bar grows one more cell inward.
- A two-pixel marker in those first two inward cells is cleared back to background.
- The cleared two-pixel marker is replaced by a two-pixel mark of the edge-bar color at the opposite side of the same lane.

Key insight
- The full edge bar defines both the orientation and the direction of the local two-step marker cycle.
</description>
</reference_puzzle>

<reference_puzzle index="17" source="eval" puzzle_id="dbff022c" lines="275-288">
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

<reference_puzzle index="18" source="train" puzzle_id="8618d23e" lines="9-25">
<description>
Stagger the two horizontal input regions onto a larger maroon separator canvas.

Input structure:
- The input is a rectangular grid with an even number of rows.
- The top half and bottom half are the two regions to preserve.
- Maroon is absent from the input and is used only as the new separator color.

Rule:
- Create an output with one extra row and one extra column, filled with maroon.
- Copy the top half into the upper-left corner.
- Copy the bottom half one row below the top region and one column to the right.

Key insight:
The output is not a recoloring of either region; it inserts an L-shaped maroon
gap by diagonally offsetting the lower half from the upper half.
</description>
</reference_puzzle>

<reference_puzzle index="19" source="train" puzzle_id="d753a70b" lines="81-93">
<description>
Red diamond outlines contract by one layer and gray diamond outlines expand by one layer while other objects stay fixed.

Input structure
- A 16x16 orange-background grid contains one-cell-thick Manhattan-diamond outlines.
- Red and gray outlines are the changing objects; other colored diamond outlines are static.
- Some outlines may be clipped by the grid boundary, but their visible pixels still fit a clipped Manhattan-distance contour.
Rule
- Clear all red and gray pixels.
- Red objects are redrawn with the same diamond center and radius minus one.
- Gray objects are redrawn with radius plus one; an ambiguous edge-clipped growing arc continues away from the nearest perpendicular edge.
Key insight
- The task is not about moving whole objects: each changing object is a contour of a diamond, and the visible operation is one step of contour contraction or expansion.
</description>
</reference_puzzle>

<reference_puzzle index="20" source="train" puzzle_id="d749d46f" lines="32-51">
<description>
Pack each solid foreground rectangle twice: wide across the top and tall from the bottom.

Input structure
The grid has one background color and one foreground color; the foreground forms
separate solid rectangles, ordered from left to right.

Rule
Build a 10-row background canvas whose width is the packed width of the wide
versions of the rectangles. For each source rectangle, paint a top-aligned
rectangle of shape min(h,w) by max(h,w), with one background column between
consecutive rectangles. Then, starting again from the left edge, paint a
bottom-aligned rectangle of shape max(h,w) by min(h,w) for each source rectangle,
again separated by one background column.

Key insight
The input rectangles are not copied in their original orientations; each object
contributes one horizontal-facing and one vertical-facing normalized rectangle,
preserving left-to-right object order.
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
   - **不建议使用**（这些是算法量而非视觉量，会迫使解题者依赖公式而非视觉直觉）：曼哈顿 / 欧氏 / Chebyshev 距离作为精确查表键、凸包面积、二进制位编码 / 位运算、模运算 / 余数、拓扑距离（BFS / 最短路 / 跳数）、精确几何质心坐标、凸角 / 凹角计数等复杂且不易被看出来的计算。
3. 问题必须能从少量示例中推断出来（通常为 3 对训练样例 + 1 个测试样例）。设计时从人类解题者的视觉角度思考：规则涉及的所有关键要素（对象、关系、属性）必须在网格上肉眼可见，不应依赖人眼无法直接观察到的隐含信息（例如颜色id是不可用的）或需要推算的隐藏属性。规则背后的逻辑本身可以有一定挑战性，但解题所需的所有线索必须能从网格中直接获取。
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
  网格尺寸合理（≤30×30），使用颜色不超过 10 种，规则复杂度适中，人类能从 3-5 个样例中发现规律。
