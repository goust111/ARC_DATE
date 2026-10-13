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

<reference_puzzle index="2" source="eval" puzzle_id="78332cb0" lines="10-23">
<description>
Reorder 5×5 blocks into a connected strip by matching edge foreground pixels.

Input structure:
- bg = background; sep = divider color (full row/column lines).
- Several 5×5 sprite blocks in a grid, separated by sep lines.

Rule:
- Two blocks "connect" when their facing edges share foreground pixels at the same offset (e.g., A's right edge ∩ B's left edge).
- Chain all blocks in the orientation (horizontal or vertical) that admits a complete sequence.
- Output is a 1×N strip of blocks in chain order, separated by sep lines.

Key insight: Edge-contact overlap determines both block ordering and output orientation.
</description>
</reference_puzzle>

<reference_puzzle index="3" source="train" puzzle_id="a644e277" lines="31-50">
<description>
Extract the lattice subregion bounded by the four background-colored corner holes.

Input structure:
- A regular one-cell-thick lattice has one line color, one background color,
  and sparse marker pixels.
- Exactly four lattice-line intersections are background-colored holes instead
  of line-color pixels.

Rule:
- Find the unique non-background color that forms regular horizontal and
  vertical lattice lines.
- Use the four background-colored lattice intersections as the corners of one
  rectangle.
- Return the inclusive crop between those corner rows and columns.

Key insight:
- The crop is marked by the missing lattice intersections, not by the marker
  pixels inside the cells.
</description>
</reference_puzzle>

<reference_puzzle index="4" source="train" puzzle_id="a6953f00" lines="45-63">
<description>
Extract the top-left or top-right 2x2 corner crop selected by the top edge clue.

Input structure
- The input is a square color grid whose answer is one of the two 2x2 crops
  touching the top edge and a grid corner.
- A repeated color on the top row, when present, acts as the primary side cue.
- Otherwise, the two top corner colors have different whole-grid frequencies.

Rule
- If a top-row color repeats, find the repeated top-row occurrence nearest to
  a left or right edge and take the 2x2 top-corner crop on that side.
- If the top row has no repeated color, compare the whole-grid frequencies of
  the two top-corner colors and take the side whose corner color is rarer.

Key insight
- The task is not asking for an arbitrary 2x2 search: the decisive information
  is on the top edge, and it selects which top corner crop to keep.
</description>
</reference_puzzle>

<reference_puzzle index="5" source="eval" puzzle_id="8698868d" lines="9-23">
<description>
Match sprites to frames by connected-hole count, then nest each sprite inside its frame.

Input structure:
- bg = azure (8). Large hollow squares (frames) arranged in a grid; smaller filled squares (sprites) placed elsewhere.
- Each frame and sprite has a distinct color. Sprites may contain holes (non-sprite-color pixels).

Rule:
- Frames form a regular grid; some may be bg-colored (invisible).
- Count connected hole regions inside each frame's interior and each sprite.
- Pair each frame with the sprite sharing the same hole count.
- Place the matched sprite inside the frame border; recolor sprite holes to the frame's color.

Key insight: The matching key is the number of connected hole components, not hole area or color.
</description>
</reference_puzzle>

<reference_puzzle index="6" source="train" puzzle_id="c4d1a9ae" lines="28-48">
<description>
Fill each vertical foreground band's background cells with the color of the next band to its right.

Input structure
- The dominant color is the background.
- Each other color occupies a separate vertical band defined by that color's
  minimum and maximum columns.
- Within a band, all cells are either that band's color or background.

Rule
- Order the bands from left to right.
- In each band, replace background pixels with the color of the next band in
  that order, wrapping from the rightmost band back to the leftmost.
- Preserve all existing foreground pixels and all background pixels outside
  the bands.

Key insight
- The column bands form a cyclic palette legend: a band's gaps are completed
  by borrowing the color from its right-hand neighbor, not by changing the
  band's existing pixels.
</description>
</reference_puzzle>

<reference_puzzle index="7" source="train" puzzle_id="df9fd884" lines="33-47">
<description>
Move the colored object from the lower bracket lane to the upper bracket lane and drop it to the bottom edge.

Input structure
- The background is the dominant color.
- One frame color forms exactly two 4-connected bracket objects: a top-edge bracket open downward and a bottom-edge bracket open upward.
- One other 4-connected foreground object sits in the lower bracket's interior column span.

Rule
- Remove the movable object from its original cells.
- Keep its shape and color unchanged.
- Paste it with its bbox columns aligned to the top bracket's interior columns and with its bbox bottom aligned to the grid bottom.

Key insight
- The lower bracket identifies the object's source lane; the upper bracket provides the target lane, and the final vertical placement is bottom-aligned.
</description>
</reference_puzzle>

<reference_puzzle index="8" source="train" puzzle_id="e048c9ed" lines="11-26">
<description>
Add a quantity-color label in the top marker column for each horizontal bar.

Input structure
- The grid has a black background and one non-background marker pixel in the top row.
- Below the marker row, every non-empty source row contains one solid monochrome horizontal bar to the left of the marker column.
- Source rows are staggered every other row.

Rule
- Copy the input grid.
- For each source bar, read its quantity as the bar length.
- Paint the quantity color in the marker column on the same row as the bar:
  2->1, 3->4, 4->9, 5->6.

Key insight
- The grey marker supplies the result column, while each bar supplies a small quantity with a fixed color code.
</description>
</reference_puzzle>

<reference_puzzle index="9" source="train" puzzle_id="ad173014" lines="97-111">
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

<reference_puzzle index="10" source="train" puzzle_id="aa62e3f4" lines="18-35">
<description>
Replace the outlined object with the outside contour of that outline in the marker color.

Input structure:
- A dominant background surrounds one 8-connected foreground outline.
- The outline is the unique largest foreground color.
- One or more smaller interior colors are inside the outline; the marker color
  is the unique smallest non-outline color.

Rule:
- Erase all original foreground pixels to background.
- Paint every background pixel that is 4-adjacent to the outline with the
  marker color.
- Keep the grid shape unchanged.

Key insight: The colored interior marker supplies the paint color, while the
largest outline supplies only the contour whose outside edge is traced.
</description>
</reference_puzzle>

<reference_puzzle index="11" source="eval" puzzle_id="136b0064" lines="25-39">
<description>
Draw a polyline by decoding 3x3 glyph instructions read across a divided grid.

Input structure:
- Left of a full-height vertical divider: 3x3 colored glyphs on bg.
- Right of divider: a single gray (5) start pixel.

Rule:
- Remove the divider; shift the gray start into the left canvas.
- Read glyphs in column-major order (left→right, top→bottom).
- Each glyph's shape (pixel area) encodes a segment: area 4 → down 2, area 5 → right 3, area 6 → left 4, area 7 → left 2.
- Append segments sequentially, stepping down one cell between each.

Key insight: Glyph pixel count (area) uniquely determines both segment length and direction — the shape itself is the instruction.
</description>
</reference_puzzle>

<reference_puzzle index="12" source="train" puzzle_id="6d1d5c90" lines="8-23">
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

<reference_puzzle index="13" source="train" puzzle_id="a1aa0c1e" lines="74-97">
<description>
Compress the stacked colored bars into a compact ordered table with the control pixel on the selected row.

Input structure
- A background grid contains several full-width colored bars from top to
  bottom; the last bar is the base/separator color.
- Each non-base bar may have an aligned three-column comb motif below it,
  alternating side-pair rows with solid rows.
- A singleton control pixel sits away from the colored motifs.

Rule
- Make one output row per non-base bar, preserving top-to-bottom order.
- In that row, paint the bar color once for each solid three-pixel row in
  its motif, then paint the base-bar color in the separator column.
- The singleton control pixel sits immediately above the base bar. Among
  motifs whose open side-pair parity is compatible with that bottom row,
  put the control pixel in the output row with the fewest solid rows; use
  horizontal distance to the motif center only to break count ties.

Key insight
- The vertical comb height is a count, while the control pixel is read by
  projecting it onto the center gap of compatible comb rows rather than by
  its vertical band alone.
</description>
</reference_puzzle>

<reference_puzzle index="14" source="eval" puzzle_id="e87109e9" lines="68-83">
<description>
Emit four teal beams from a square, turning at colored obstacles per a legend.

Input structure:
- Top band: grey (5) legend with colored vertical bars in cells; bar position (left or right) encodes turn direction.
- Below: main grid with bg, a solid teal (8) square, and colored obstacle blocks.

Rule:
- Remove the legend from the output.
- From the teal square, shoot a beam in each cardinal direction (as a sliding copy of the square).
- Each beam paints teal over bg cells as it moves.
- On hitting an obstacle of color C, consult the legend: bar on left → turn counterclockwise, bar on right → turn clockwise. Continue in the new direction.
- Beams stop at grid edges.

Key insight: The legend's bar position (left vs right within its cell) is the encoding for turn direction at each obstacle color.
</description>
</reference_puzzle>

<reference_puzzle index="15" source="train" puzzle_id="afe3afe9" lines="106-119">
<description>
Convert each hollow 3x3 symbol to one pixel and slide the pixels clockwise from the marked edge.

Input structure
- The grid has a background, one full-edge marker line, and monochrome hollow 3x3 foreground symbols.
- The symbols lie on a coarse row/column lattice; fully empty lattice rows or columns are absent from the output.

Rule
- Replace each hollow 3x3 symbol with one pixel of its color in the corresponding coarse-grid cell.
- The marker edge gives the compaction direction by clockwise border travel: top -> right, right -> down, bottom -> left, left -> up.
- Preserve the order of non-background pixels while sliding them as far as possible in that direction.

Key insight
- The large picture is a scaled symbolic grid, and the blue edge marker acts as a gravity direction rather than as output content.
</description>
</reference_puzzle>

<reference_puzzle index="16" source="train" puzzle_id="f8cc533f" lines="10-25">
<description>
Complete every colored sprite by applying the union of all aligned partial sprite masks.

Input structure:
- The dominant color is the background.
- Each non-background color appears as a separated partial copy of one common sprite mask.
- The copies share a top-left-aligned sprite window; some copies may be missing lower or right edge pixels.

Rule:
- Crop each same-color mask to its bounding box.
- Pad the crops to the maximum observed height and width, aligned at the top-left, and OR them into one complete template.
- Fill that template at every sprite location using the sprite's own color.

Key insight:
- No single input sprite is complete, but their missing cells are complementary, so the aligned union reveals the full mask.
</description>
</reference_puzzle>

<reference_puzzle index="17" source="train" puzzle_id="60d73be6" lines="22-39">
<description>
Complete the sparse colored marker pattern by reflecting it across a full-row/full-column cross.

Input structure
The grid has a dominant background and one non-background cross color that
fills exactly one complete row and one complete column. All other foreground
pixels are sparse markers away from that cross.

Rule
Keep the input unchanged, and for each marker pixel copy its color to the
positions obtained by reflecting it across the cross row, across the cross
column, and across both axes. Reflected positions outside the grid are
clipped.

Key insight
The full cross is not a copied object; it is the pair of mirror axes for
completing the quadrant pattern.
</description>
</reference_puzzle>

<reference_puzzle index="18" source="eval" puzzle_id="7c66cb00" lines="9-24">
<description>
Dock two-tone sprites into color-matched horizontal bars with a color swap.

Input structure:
- bg = most common color.
- Top area: small two-tone sprites (outer color + inner color).
- Below: horizontal bars spanning most of the row, each with a fill color and border color on edges.

Rule:
- Match each sprite to the bar whose fill color equals the sprite's outer color.
- Place the sprite at its original x-position, aligned to the bar's bottom.
- Recolor: sprite outer → bar border, sprite inner → bar fill.
- Erase original sprites from the top area.

Key insight: Color matching is inverted — the sprite's dominant color identifies its target bar, then gets swapped to the bar's accent color.
</description>
</reference_puzzle>

<reference_puzzle index="19" source="train" puzzle_id="880c1354" lines="14-28">
<description>
Recolor the separated outer regions by rotating their colors one step clockwise around the yellow-orange divider.

Input structure:
- The grid contains a fixed yellow/orange divider structure.
- All other colors form separated 4-connected monochrome regions around that divider.

Rule:
- Leave the yellow and orange pixels unchanged.
- Sort the non-divider regions by their centroid angle around the divider centroid in screen-coordinate clockwise order.
- Each region keeps its cells and receives the color of the previous region in that cyclic order.

Key insight:
- The objects do not move; only the color labels cycle through the spatial order of the regions around the divider.
</description>
</reference_puzzle>

<reference_puzzle index="20" source="eval" puzzle_id="f931b4a8" lines="10-27">
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

