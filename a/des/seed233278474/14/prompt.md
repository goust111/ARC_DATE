你是一个 ARC-AGI 谜题设计专家。下面的 20 道参考题是难度和机制质量的标杆。仔细阅读每道题的**解题逻辑**（解题者从样例归纳规则的认知过程），而非表面场景。你的新题应在机制深度和隐蔽性上与参考题持平或者实现超越。

每个参考题描述包含：
- `description`：一句话概述题目的变换目标。
- `Input structure`：输入网格中有哪些对象、背景、标记、边界或组件。
- `Rule`：从输入到输出的确定性变换规则。
- `Key insight`：解题所需的核心洞见。

参考题目如下：
<reference_puzzle index="1" source="eval" puzzle_id="16de56c4" lines="9-34">
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

<reference_puzzle index="2" source="eval" puzzle_id="9385bd28" lines="71-87">
<description>
Fill rectangles between diagonal marker pairs using a color legend.

Input structure:
- bg grid with a vertical legend at the bottom-left: each row is a (marker_color, fill_color) pair.
- Scattered L-shaped corner markers of each marker color define opposite corners of axis-aligned rectangles.
- Some markers are solid 2×2 blocks instead of corner pairs.

Rule:
- Read legend bottom-to-top; for each entry find all pixels of that marker color.
- Determine which two bbox corners carry L-shaped anchors — the *other* diagonal defines the fill region.
- Fill the diamond (all shortest-path cells between those two corners) with the legend's fill color.
- Solid 2×2 marker blocks are simply recolored to the fill color.
- Later legend entries are protected from earlier fills.

Key insight: The L-shaped anchors sit on the marker diagonal; the fill goes on the opposite diagonal of the bounding box.
</description>
</reference_puzzle>

<reference_puzzle index="3" source="train" puzzle_id="b5bb5719" lines="17-32">
<description>
Fill the background under a top-row red/grey span with a shrinking diagonal pattern that alternates red and grey.

Input structure:
- The grid has an orange background.
- All foreground pixels are red or grey and lie on the top row.
- The first and last top-row foreground pixels define an odd-width span.

Rule:
- Copy the input grid.
- For row r below the top, fill columns span_left + r through span_right - r.
- Each filled cell takes the top-row color r columns up-left, flipped red<->grey once per row step.

Key insight:
- The output is the top-row span propagated down-right along diagonals, while the allowable region shrinks inward from both ends each row.
</description>
</reference_puzzle>

<reference_puzzle index="4" source="eval" puzzle_id="cb2d8a2c" lines="13-26">
<description>
Extend a green path from one grid edge to the opposite, routing around red bars at a distance dictated by their blue pixel count.

Input structure:
- bg = azure (8). Red/blue bars (color 2/1) act as obstacles.
- A single green (3) pixel sits on a grid border — the path start.

Rule:
- Each bar's keep-out radius equals its number of blue pixels; the path must stay farther than that from every bar cell.
- The green pixel extends inward toward the opposite edge, turning at right angles to navigate around forbidden zones.
- After the path is drawn, all blue pixels become red.

Key insight: Blue pixels on a bar encode its danger radius — more blue means a wider no-go zone the path must respect.
</description>
</reference_puzzle>

<reference_puzzle index="5" source="train" puzzle_id="5d588b4d" lines="10-28">
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

<reference_puzzle index="6" source="train" puzzle_id="da6e95e5" lines="70-81">
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

<reference_puzzle index="7" source="train" puzzle_id="5adee1b2" lines="8-23">
<description>
Use the left-edge color legend to draw a padded rectangular halo around each matching object while preserving enclosed holes.

Input structure:
- The grid has a dominant background color.
- One or more two-row markers sit on the left edge; each marker is a 2x2 block whose left column is an object color and right column is its halo color.
- Away from the markers, every source color from the legend appears as one or more 4-connected monochrome objects.

Rule:
- Keep the input pixels unchanged.
- For each source-color object, take its tight bbox expanded by one cell and clipped to the grid.
- Recolor background pixels in that expanded region to the marker's halo color, except for background holes enclosed by the object.

Key insight:
- The small left-edge bars are a legend: they map object colors to the colors used for the surrounding padded bbox.
</description>
</reference_puzzle>

<reference_puzzle index="8" source="train" puzzle_id="6d1d5c90" lines="8-23">
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

<reference_puzzle index="9" source="eval" puzzle_id="f931b4a8" lines="10-27">
<description>
Count the two upper quadrant control regions to size an output made from the lower quadrant pattern.

Input structure:
- The grid is an even square split into four equal quadrants.
- Black is the background.
- The upper-left and upper-right quadrants are control regions.
- The lower-left quadrant is a color source tile; the lower-right quadrant is either a stencil tile or empty.

Rule:
- Output height is the number of non-background pixels in the upper-left quadrant.
- Output width is the number of non-background pixels in the upper-right quadrant.
- If the lower-right stencil has foreground pixels, tile it over the output; each stencil background pixel is filled from the lower-left quadrant cell at the corresponding coarse tile position.
- If the lower-right stencil is empty, tile the lower-left quadrant directly over the output.

Key insight:
- The top quadrants specify only dimensions, while the bottom quadrants specify a two-level repeated pattern: a fine stencil and coarse fill colors.
</description>
</reference_puzzle>

<reference_puzzle index="10" source="train" puzzle_id="9f669b64" lines="26-48">
<description>
Push the central solid rectangle to the grid edge, splitting the perpendicular target rectangle around its path.

Input structure
- The background is the dominant color.
- There are exactly three monochrome foreground objects arranged on one
  horizontal or vertical axis.
- The middle object on that axis is a solid rectangular moving object.
- One end object is a solid rectangle or bar whose long side is perpendicular
  to the movement axis; this is the target to split.

Rule
- Move the middle rectangle along the axis toward the target until it is
  flush with the grid edge in that direction.
- Clear the original middle rectangle and the original target.
- Split the target through the moving rectangle's two-cell-wide path and
  shift the two halves one cell outward perpendicular to the movement.
- Keep the other end object unchanged.

Key insight
- The non-target end object acts as a pusher; the central rectangle travels
  away from it through the perpendicular target, opening a matching gap.
</description>
</reference_puzzle>

<reference_puzzle index="11" source="train" puzzle_id="e4888269" lines="67-88">
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

<reference_puzzle index="12" source="eval" puzzle_id="faa9f03d" lines="36-52">
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

<reference_puzzle index="13" source="train" puzzle_id="15660dd6" lines="33-54">
<description>
Combine each column's repeated template mask with its single solid color swatch.

Input structure:
- The grid is a table of framed panels separated by background rows/columns.
- The leftmost column gives one solid row-label color for each panel row.
- In each panel column, exactly one panel has a solid non-background interior,
  while the other panels share the same mask drawn in one template color on
  the background.

Rule:
- For every panel column, find the row containing the solid interior swatch.
- Rebuild one output panel using that row's left label as the frame color.
- Fill the repeated template mask with the swatch color and leave the mask
  background cells as background; separate rebuilt panels with one
  background column.

Key insight:
The table encodes a colorized stencil per column: the solid panel supplies
the paint color and frame label, while the duplicate template panels supply
the shape mask.
</description>
</reference_puzzle>

<reference_puzzle index="14" source="eval" puzzle_id="6ffbe589" lines="83-98">
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

<reference_puzzle index="15" source="eval" puzzle_id="a32d8b75" lines="262-275">
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

<reference_puzzle index="16" source="train" puzzle_id="9720b24f" lines="34-46">
<description>
Delete each foreign-color object whose bounding box is contained by a larger object of another color.

Input structure
- The black pixels are the empty canvas, and every other color forms one or more 8-connected monochromatic objects.
- Some larger foreground objects act as loose U/C-like containers around smaller objects of different colors.
Rule
- Find all 8-connected monochromatic non-black objects.
- Recolor an object to black exactly when its bounding box is contained in the bounding box of a larger object of another color.
- Keep all other pixels unchanged.
Key insight
- The removable objects are identified by containment in another color's rectangular extent, not by their own color or shape.
</description>
</reference_puzzle>

<reference_puzzle index="17" source="eval" puzzle_id="2c181942" lines="7-21">
<description>
Attach monochrome sprites to the matching sides of a central hollow square.

Input structure:
- bg = background. A hollow NxN square (hub) with bg corners/interior and four differently-colored edges (top, bottom, left, right).
- 2-4 monochrome sprites scattered on the grid, each matching one hub edge color.

Rule:
- For each hub edge color, find the matching sprite outside the hub.
- Rotate the sprite (0/90/180/270°) so its contact edge has exactly N-2 contiguous cells aligned with the hub side, and the rotated sprite is mirror-symmetric along the contact-edge axis (UD for left/right, LR for top/bottom).
- Translate it to adjoin that hub side. 
- Clear everything else to bg; output is hub + attached sprites only.

Key insight: Each edge color uniquely pairs a sprite to a side; symmetry along the contact-edge axis disambiguates the correct rotation.
</description>
</reference_puzzle>

<reference_puzzle index="18" source="train" puzzle_id="f0f8a26d" lines="30-44">
<description>
Rotate every foreground bar 90 degrees around its own center on the same canvas.

Input structure
The background is the dominant color, and all foreground pixels share one color.
Each 4-connected foreground object is a solid one-cell-thick horizontal or vertical
bar with odd length, with single-pixel bars allowed.

Rule
Clear the canvas to the background, then redraw every bar with its midpoint, length,
and color preserved while swapping horizontal and vertical orientation.

Key insight
The bars are independent objects: their centers and lengths identify the output
placement, so no matching, ordering, or global symmetry is needed.
</description>
</reference_puzzle>

<reference_puzzle index="19" source="eval" puzzle_id="31f7f899" lines="7-21">
<description>
Sort vertical bars by height and redistribute them across fixed colored slots.

Input structure:
- bg background; a horizontal base line of a single color spans the full grid width.
- Several monochrome vertical bars of distinct colors rise upward from the base line.

Rule:
- Rank bar shapes by height (shortest to tallest).
- Rank original x-positions (slots) left to right.
- Place the i-th shortest shape into the i-th leftmost slot, recoloring it to that slot's original color.
- The base line is preserved unchanged.

Key insight: Each slot keeps its x-position and color identity; only the bar silhouettes are redistributed among slots by height order.
</description>
</reference_puzzle>

<reference_puzzle index="20" source="train" puzzle_id="4df5b0ae" lines="27-41">
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
