你是一个 ARC-AGI 谜题设计专家。下面的 20 道参考题是机制质量和隐蔽性的标杆。仔细阅读每道题的**解题逻辑**（解题者从样例归纳规则的认知过程），而非表面场景。本轮只生成**多层结构**的新题（判定标准见注意事项 9），机制深度应与参考题中最难的一档持平或实现超越。

每个参考题描述包含：
- `description`：一句话概述题目的变换目标。
- `Input structure`：输入网格中有哪些对象、背景、标记、边界或组件。
- `Rule`：从输入到输出的确定性变换规则。
- `Key insight`：解题所需的核心洞见。

参考题目如下：
<reference_puzzle index="1" source="train" puzzle_id="e4941b18" lines="8-22">
<description>
Move red into teal's top marker position, then move teal to the bottom outside corner of the grey rectangle.

Input structure
- The dominant background surrounds one filled rectangular non-background block.
- There is exactly one red marker and one teal marker, both in the row immediately above the rectangle.
- Both markers are horizontally aligned over columns inside the rectangle.

Rule
- Clear the original red marker.
- Recolor the original teal marker cell red.
- Place the teal marker beside the rectangle's bottom row, on the side indicated by the horizontal direction from red to teal.

Key insight
- The red-to-teal horizontal order chooses the rectangle side where teal exits at the bottom.
</description>
</reference_puzzle>

<reference_puzzle index="2" source="train" puzzle_id="a416fc5b" lines="95-106">
<description>
Add the two missing colored arms of a centered diamond-sprite motif, or clear a completed motif to a blank square.

Input structure
An 11x11 board has orange background and pink separator bars forming nine 3x3 cells. Occupied cells contain a four-pixel diamond sprite; the center cell and one neighboring cell are red, and a completed board may also contain one grey and one teal sprite.

Rule
With only the two red sprites present, preserve the board and add grey and teal diamond sprites 135 degrees clockwise and counterclockwise from the non-center red arm. If those expected grey and teal sprites are already present, output an all-background square whose side length is the total number of sprite pixels.

Key insight
The non-center red sprite is an eight-neighbor direction from the center; the other two colored arms are fixed rotations of that direction.
</description>
</reference_puzzle>

<reference_puzzle index="3" source="eval" puzzle_id="89565ca0" lines="120-136">
<description>
Convert each colored partitioned frame into a horizontal bar whose length is its number of enclosed regions.

Input structure:
- Black background with several axis-aligned colored line drawings.
- The line drawings are damaged by scattered pixels of one non-background noise color and by overlaps with other drawings.
- Each real drawing is a rectangular partitioned frame enclosing one or more background regions.

Rule:
- For each non-background color, reconstruct its horizontal and vertical wall segments from dense row/column evidence.
- Count the 4-connected background regions fully enclosed by those reconstructed walls.
- The color with no enclosed region is the noise/pad color.
- Sort drawing colors by enclosed-region count; equal counts are ordered by which reconstructed frame covers the other.

Key insight:
- The scattered color is not an object; it only pads the histogram, while each true object's room count is stable after rectilinear wall reconstruction.
</description>
</reference_puzzle>

<reference_puzzle index="4" source="train" puzzle_id="20fb2937" lines="63-77">
<description>
Replace each marker below the separator with the mapped top legend square centered on that marker.

Input structure:
- The background surrounds a top legend, one solid non-background separator row, and a lower marker field.
- The legend contains equal-size filled square sprites, with a singleton marker below each sprite and aligned to its center column.
- The lower field contains only background and marker colors from the legend.

Rule:
- Pair legend markers to legend sprites by left-to-right position.
- Discard the legend and separator; keep the grid width and the rows below the separator.
- In the lower field, replace every marker pixel by a filled square of the paired sprite color, centered at the marker location.

Key insight: The top area is a positional color legend, and the actual output is the post-separator marker field expanded by that legend.
</description>
</reference_puzzle>

<reference_puzzle index="5" source="train" puzzle_id="963c33f8" lines="86-108">
<description>
Drop the columns of the top marker downward, stopping blue-ended columns on grey obstacles and letting maroon-ended columns pass to the bottom.

Input structure
- The background is the dominant color.
- The most numerous non-background color forms stationary grey obstacle pixels.
- The other two colors form a single 3x3 top-edge marker whose top two rows
  are maroon and whose bottom row contains maroon or blue control pixels.

Rule
- Erase the top marker.
- Treat each marker column as a vertical drop retaining its three colors.
- A column whose bottom control pixel is blue stops immediately above the
  first grey pixel below that column, or bottom-aligns if none exists.
- Contiguous maroon-ended columns pass through obstacles and bottom-align;
  if such a run has a clear down-right slide from a grey ledge, shift the
  run one column right before bottom-aligning.
- When a maroon-ended drop lands through a grey bar on the bottom edge, clear
  the rest of the marker-width gap in that bottom bar.

Key insight
- The bottom row of the top marker is a control strip: blue cells mean
  "stick to the first obstacle," while maroon cells mean "fall through."
</description>
</reference_puzzle>

<reference_puzzle index="6" source="train" puzzle_id="22806e14" lines="23-37">
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

<reference_puzzle index="7" source="train" puzzle_id="e45ef808" lines="31-46">
<description>
Mark the shallowest and deepest columns of a solid terrain boundary with vertical colored guide lines.

Input structure:
- The first row is a solid empty separator.
- Below it, each column consists of upper fill cells followed by a contiguous suffix of terrain cells.
- The bottom row is all terrain, so every column has a well-defined first terrain row.

Rule:
- Find the column whose first terrain row is latest; it is unique and its upper fill run becomes maroon.
- Find the rightmost column whose first terrain row is earliest; its upper fill run becomes yellow.
- Preserve the empty separator, terrain cells, and all other upper fill cells.

Key insight:
- The task is a column-height summary of the terrain profile: color the deepest upper run and the highest terrain peak.
</description>
</reference_puzzle>

<reference_puzzle index="8" source="train" puzzle_id="c4d1a9ae" lines="28-48">
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

<reference_puzzle index="9" source="eval" puzzle_id="247ef758" lines="35-46">
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

<reference_puzzle index="10" source="train" puzzle_id="8abad3cf" lines="10-25">
<description>
Repack each non-background color group as a solid square ordered by increasing size.

Input structure
The grid has one dominant background color, and each other color has a
perfect-square pixel count. Those square side lengths are unique.

Rule
Convert every non-background color into a filled square whose side length is
the square root of its pixel count. Place the squares from left to right in
increasing side length, bottom-aligned, with one background column between
neighboring squares.

Key insight
The original geometry of each color is discarded; only the color's total
pixel count determines the size of its packed output square.
</description>
</reference_puzzle>

<reference_puzzle index="11" source="train" puzzle_id="465b7d93" lines="97-117">
<description>
Move the small colored sprite into the hollow frame by stretching its shape to the frame interior.

Input structure:
- The background is the dominant color.
- There are exactly two foreground colors: one forms a complete rectangular
  one-pixel-thick frame, and the other forms one 4-connected sprite outside it.
- The frame interior is initially background.

Rule:
- Clear the original sprite from its outside location.
- Map the sprite bbox to the frame interior bbox, sending bbox extremes to
  interior extremes.
- Draw straight horizontal or vertical segments between mapped positions of
  adjacent sprite pixels, then fill any enclosed background region inside
  those drawn segments.

Key insight:
The outside sprite is a miniature line/fill diagram for the frame interior,
not a bitmap to scale by block replication.
</description>
</reference_puzzle>

<reference_puzzle index="12" source="train" puzzle_id="5ecac7f7" lines="23-36">
<description>
Combine the left, middle, and right ordered objects from three separated panels.

Input structure
- The grid is three HxH panels separated by one-cell vertical separator bars.
- The background is the dominant color, and each panel contains exactly three same-color 4-connected foreground objects.

Rule
- Build a new HxH background canvas.
- From the left panel copy its leftmost object, from the middle panel copy its middle object, and from the right panel copy its rightmost object.

Key insight
- The separator bars create three aligned layers, and the panel's own left-to-right object order selects which object survives in the merged output.
</description>
</reference_puzzle>

<reference_puzzle index="13" source="eval" puzzle_id="b9e38dc0" lines="7-18">
<description>
Fill the region enclosed by a wall sprite and radiate color outward through its opening.

Input structure:
- bg = most common color. Wall = most frequent non-bg color, forming a partial enclosure (e.g. U-shape).
- A fill-color pixel sits inside the enclosed area. Other non-bg pixels may act as obstacles.
Rule:
- Flood-fill the enclosed area with the fill color, radiating outward through the wall's open end.
- Obstacles cast "shadows" along the radiate direction, blocking fill behind them.

Key insight: The wall endpoints define the opening direction; fill radiates outward through it, and obstacles block propagation along that same axis.
</description>
</reference_puzzle>

<reference_puzzle index="14" source="train" puzzle_id="2601afb7" lines="30-45">
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

<reference_puzzle index="15" source="train" puzzle_id="5adee1b2" lines="8-23">
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

<reference_puzzle index="16" source="train" puzzle_id="2e65ae53" lines="88-103">
<description>
Fill every lattice slot from the color assigned to that slot position by the other lattices.

Input structure:
- The background is the dominant color.
- The largest foreground color forms several one-pixel-thick rectangular lattice frames.
- The frame lines divide each lattice into the same row-by-column set of rectangular slots; some slots already contain one non-frame color.

Rule:
- Infer a color for each logical slot position from the colored slots across all lattice frames.
- Repaint every slot interior in every lattice with the inferred color for its row/column position.
- Keep the frame lines and all background outside frames unchanged.

Key insight:
- Slot sizes vary by frame, but the complete color pattern is indexed by lattice slot position rather than by absolute pixel size or location.
</description>
</reference_puzzle>

<reference_puzzle index="17" source="eval" puzzle_id="d8e07eb2" lines="29-47">
<description>
Look up query glyphs in a reference table and mark whether they form a line.

Input structure:
- Teal background with two full-width pink separator rows.
- Above the first separator is a query row of 5x5 glyph cells.
- Between separators is a reference table of the same 5x5 cells; a glyph is
  identified by both its color and its 3x3 shape.

Rule:
- Green-fill the background inside every table cell whose glyph exactly
  appears in the query row.
- If the matched table cells form one complete row or one complete column,
  also green-fill the query band and the bottom verdict band.
- Otherwise leave the query band unchanged and fill the bottom verdict band red.

Key insight: The bottom color is not a count; it is the yes/no answer to
whether the queried glyphs align as a full row or column in the table.
</description>
</reference_puzzle>

<reference_puzzle index="18" source="train" puzzle_id="da6e95e5" lines="70-81">
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

<reference_puzzle index="19" source="train" puzzle_id="a1aa0c1e" lines="74-97">
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

<reference_puzzle index="20" source="train" puzzle_id="342ae2ed" lines="34-49">
<description>
Draw the missing diagonal connector segment between each pair of same-color rectangular objects.

Input structure:
- The grid has one dominant background color.
- Every foreground color appears as exactly two separated solid rectangular 4-connected objects.
- The two rectangles of each color are diagonally offset with facing corners on a 45-degree line.

Rule:
- Keep all existing rectangles unchanged.
- For each foreground color, connect the lower facing corner of the upper rectangle to the upper facing corner of the lower rectangle.
- Paint only the intervening background pixels on that diagonal, excluding the two endpoint pixels already in the rectangles.

Key insight:
- Same-color rectangles define the endpoints; the output completes the open diagonal between their nearest facing corners.
</description>
</reference_puzzle>

你的任务：
以参考题为标杆，运用发散性思维进行创新扩展。请发挥创造力，构思全新且巧妙的推理谜题。用中文创造全新的 ARC 谜题描述。
本轮每道新题都必须是**多层结构**：规则包含至少两个阶段，每个阶段都需要解题者从样例中主动归纳一个非显然规律，且缺少任何一层都无法得到正确输出。生成的题目中至少有 2 道为三层及以上结构（层数 = 需要各自归纳的阶段数）。
创新方式包括但不限于：
**机制重组**：将多道参考题各自的核心机制提取出来，组合成一个新的多步管道，用到新的场景。
**机制变异**：取几道参考题的核心机制，以机制为基础进行创新并与其他多种机制进行创新重组，使解题路径发生质变而非量变。
**逆向构造**：取一道参考题的输出模式，思考"什么样的输入规则能产生类似的输出结构但需要完全不同的推理过程"。

以上不是可选建议，而是硬性要求：每道新题都必须以本轮参考题的机制为出发点（三种方式任选其一），并在 `source` 字段中标注取自哪几道参考题（写编号）的什么机制、变异点或新组合点在哪。允许单个机制与某道参考题相同，但组合后的整体推理路径必须是新的。

严格使用下面的 XML 风格格式输出：

<new_puzzle>
- `description`：一句话概述题目的变换目标。
- `Input structure`：输入网格中有哪些对象、背景、标记、边界或组件。
- `Rule`：从输入到输出的确定性变换规则。
- `Key insight`：解题所需的核心洞见。
- `source`：取自参考题 X、Y 的什么机制，变异点/新组合点是什么。
</new_puzzle>

请一次性生成5个`<new_puzzle>`，若有较多合适的创新，也可以生成更多，不设上限，但不要输出额外解释。

注意事项：
1. 任务不应依赖大量先验知识或复杂计算，而应依赖对模式、结构与空间关系的直觉推理。
2. 关于"计算"的边界：
   - **允许使用**（这些都是人类能心算的视觉量）：数像素 / 孔洞 / 对象 / 邻居（≤10 量级）、比大小、按某个可见属性排序、奇偶判断、取最大 / 最小、用一个计数值作为另一个等量几何量（如 N 个像素 → 长度 N 的射线、面积 N → 边长 N）、判断接触 / 相邻 / 包含、粗略中心位置（簇的视觉中点）、"距离"作为"几步"而不是公式等人类能够直观感受计算的内容。
   - **禁止使用**（这些是算法量而非视觉量，会迫使解题者依赖公式而非视觉直觉）：曼哈顿 / 欧氏 / Chebyshev 距离作为精确查表键、凸包面积、二进制位编码 / 位运算、模运算（取模） / 余数、拓扑距离（BFS / 最短路 / 跳数）、精确几何质心坐标、凸角 / 凹角计数等复杂且不易被看出来的计算都是禁止使用的。
3. 问题必须能从少量示例中推断出来（通常为 3 对训练样例 + 1 个测试样例）。设计时从人类解题者的视觉角度思考：规则涉及的所有关键要素（对象、关系、属性）必须在网格上肉眼可见，不应依赖人眼无法直接观察到的隐含信息（例如颜色id和颜色亮度是禁止使用的）或需要推算的隐藏属性。规则背后的逻辑本身可以有一定挑战性，但解题所需的所有线索必须能从网格中直接获取。
4. 生成的问题描述旨在促进后续用于创建题目网格的代码生成。因此，对输入 / 输出网格的描述以及规则总结必须极其详细；语言风格应易于理解；对于特定对象（如箭头），必须在描述中明确其绘制方式，但注意描述需正确且符合该对象作用，例如箭头需要有指向性，至少为三像素。
5. 描述谜题时避免使用 0 或 1 这类颜色编号；应使用颜色名称，例如 "red"。可用颜色如下：Black、Blue、Red、Green、Yellow、Grey、Magenta、Orange、Azure、Maroon。
6. 新题的核心推理机制应明显不同——即从观察样例到归纳规则的认知路径不同，而不仅仅是场景、对象或颜色不同。
7. 每一层的难度应来自机制的隐蔽性而非步骤数。用参考题作为难度标尺——如果你生成的题一个样例就能猜到规则，说明机制太透明，需要加深隐蔽性。
8. 优先从参考题的机制出发进行重组或推演，而非从物理隐喻出发。物理隐喻容易导致大量题目归结为一种模式，例如"沿方向移动到阻挡处停止"。
9. 关于"结构层数"的判定标准：**只有解题者需要主动推理才能完成的步骤才算一层**。肉眼一扫即可获得的信息（如"找到缺口在哪一侧"、"数出像素个数"、"识别哪个对象最大"）不构成独立的推理层——它们只是变换的前置输入，不增加认知难度。真正的多层结构是指每一层都需要解题者发现一个非显然的规律。以两层结构为例：
   - ✗ 伪两层："找到缺口方向"→"把东西推到对面" — 第一步肉眼可见，实际只有一层推理。
   - ✓ 真两层："通过孔洞数量匹配颜色图例"→"用匹配到的颜色重绘对象" — 两步都需要非平凡的归纳。

   本轮每道新题都必须是真多层。**用堆叠 tie-break 规则或复杂数学公式来"加层"是劣解，伪多层视为不合格**。
10. 在符合 ARC 题目和提示词要求的基础上，鼓励三层或四层结构，但不可为了层数违背要求。层与层之间应有依赖——前一层归纳的产物是后一层变换的输入，而不是两条互不相关的规则简单拼在一起。
11. 警惕**层塌缩**：每一层的判断依据应与被变换的对象分离（位于网格的其他位置，或来自跨对象的全局关系）。如果解题者对任意单个对象做一次输入输出对比，就能同时读出所有层的规则（例如线索就写在被变换对象自己身上），那么无论描述里写了几步，该题实际只有一层。

生成后对生成的题执行以下五项审查（审查内容不需要保存到文件）：
1. 原创性与取材
  单个机制允许与参考题相同，但新题的整体机制组合与核心推理路径不得与任何一道参考题相同，也不可只是替换颜色、方向、对象类型等表层要素。同时核对 `source` 字段：每道题都标注了来源参考题，且取材与变异点描述属实。
2. 确定性
  规则完全确定，同一输入只能产生唯一输出，每一步操作都有明确定义。
3. 可实现性
  所有必要细节都已给出——对象如何生成、如何识别、如何变换、边界情况如何处理，读完描述后可以直接写代码。
4. ARC 合规性
  网格尺寸合理（≤30×30）。逐题清点描述中出现的所有颜色名：必须全部来自注意事项 5 的清单（清单之外的颜色名一律不可用），且含背景在内总数不超过 10 种。规则复杂度适中，人类能从 3-5 个样例中发现规律。没有使用上述注意事项第二条禁止使用的复杂计算。
5. 层级审查
  逐题逐层检查：每一层是否都需要解题者主动归纳？若某一层肉眼一扫即得，该题为伪多层，必须加深该层或重写整题。再按注意事项 11 检查层塌缩：一次输入输出对比是否会同时暴露所有层（若是则不合格）。最后核对全批中三层及以上的题目是否达到 2 道。
