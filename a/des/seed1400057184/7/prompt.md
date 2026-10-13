你是一个设计 ARC-AGI（抽象与推理语料库，Abstraction and Reasoning Corpus）谜题的专家。你的任务是对下方提供的谜题逻辑与格式进行深入分析。通过参考示例谜题的推理与概念，你需要运用发散性思维进行创新扩展。请发挥创造力，构思全新且巧妙的推理谜题。
下面给出 20 个已有 ARC 题目的描述作为参考。每个描述通常包含：
- `description`：一句话概述题目的变换目标。
- `Input structure`：输入网格中有哪些对象、背景、标记、边界或组件。
- `Rule`：从输入到输出的确定性变换规则。
- `Key insight`：解题所需的核心洞见。

参考题目如下：
<reference_puzzle index="1" source="eval" puzzle_id="7b5033c1" lines="7-21">
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

<reference_puzzle index="2" source="eval" puzzle_id="64efde09" lines="68-83">
<description>
Copy colored control-ray ports from one rectangular sprite orientation to every matching transformed sprite.

Input structure:
- bg is the most common color.
- Several dense 2-by-N or N-by-2 rectangular sprites are dihedral transforms of one template.
- Isolated single-pixel control markers sit on rays projected outward from the dominant long edge of some sprites.

Rule:
- For each sprite, the active edge is the long side whose cells have the strongest same-color majority.
- A control marker seen first along an outward ray from an active-edge cell assigns its color to that template cell.
- On every transformed sprite, draw that assigned color outward from the corresponding active-edge cell through bg cells until blocked or out of bounds.

Key insight:
- The marker positions are not absolute; they label cells of a shared rectangular template, then the labels follow each sprite's rotation/reflection.
</description>
</reference_puzzle>

<reference_puzzle index="3" source="train" puzzle_id="b2bc3ffd" lines="8-23">
<description>
Move every object above the solid bottom stripe upward by its own pixel count.

Input structure
The grid has a dominant background, exactly one full-width non-background row
that acts as a fixed stripe, and one or more 4-connected monochrome objects
above that stripe.

Rule
Clear each object from its original cells, then translate it straight upward
by the number of pixels in that object, preserving its shape and color. Leave
the full-width stripe unchanged.

Key insight
Each object's area is also its vertical displacement; the full-width row is a
fixed baseline, not a moving object.
</description>
</reference_puzzle>

<reference_puzzle index="4" source="train" puzzle_id="df9fd884" lines="33-47">
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

<reference_puzzle index="5" source="train" puzzle_id="95755ff2" lines="45-60">
<description>
Paint straight rays from colored side markers into the hollow diamond frame until an interior foreground pixel blocks each ray.

Input structure
- The background contains one complete one-pixel-thick diamond frame.
- Colored marker pixels outside the diamond sit on the top, bottom, left, or right side of the frame's bounding box.
- Any non-background, non-frame pixels inside the diamond are blockers that remain unchanged.

Rule
- Each outside marker emits inward perpendicular to its side of the diamond bounding box.
- Along that row or column, paint only background pixels strictly inside the diamond with the marker color.
- Stop a ray when it reaches an existing interior foreground pixel; keep the frame and all original colored pixels unchanged.

Key insight
- The diamond frame defines a strict interior, while the side markers are column or row emitters whose color fills the visible interior interval until blocked.
</description>
</reference_puzzle>

<reference_puzzle index="6" source="train" puzzle_id="fc10701f" lines="20-35">
<description>
Move the orange rectangle onto the maroon rectangle and mark gated corridor crossings in red.

Input structure
- The dominant background contains black obstacle segments.
- There is one solid orange rectangle and one solid maroon rectangle of the same size.
- The two rectangles share either their row span or their column span, so the move is straight and axis-aligned.

Rule
- Erase the original orange rectangle to the background.
- Replace the maroon rectangle with the orange rectangle.
- Between the two rectangles, paint red across the moving rectangle's width or height only at corridor cells flanked by black obstacle segments perpendicular to the movement.

Key insight
- The black segments define gates in the straight corridor; only those gated openings, not every swept background cell, become red.
</description>
</reference_puzzle>

<reference_puzzle index="7" source="train" puzzle_id="3cd86f4f" lines="6-18">
<description>
Shear the rectangular input down-left onto a black canvas.

Input structure
A rectangular grid of colored pixels with no internal empty region to infer.

Rule
Create a black output with the same height and width W + H - 1, then copy
each input row r unchanged starting at column H - 1 - r.

Key insight
The colors and row contents are preserved; only the row offsets change so
the left edge forms a descending diagonal.
</description>
</reference_puzzle>

<reference_puzzle index="8" source="train" puzzle_id="e39e9282" lines="73-81">
<description>
Project maroon markers into adjacent grey or pink 3x3 squares, erasing grey squares afterward.

Input structure
The grid has a dominant background, solid 3x3 grey and/or pink squares, and maroon marker pixels or segments directly touching one side of exactly one square.
Rule
Grey squares are erased; their adjacent maroon markers stay in place and stamp the nearest edge cell inside the grey square. Pink squares stay pink; their adjacent maroon markers disappear and stamp the corresponding cell on the square's middle row or middle column.
Key insight
The side touched by a marker determines the projection: grey uses the near edge, while pink uses the centerline, preserving the marker's coordinate along that side.
</description>
</reference_puzzle>

<reference_puzzle index="9" source="train" puzzle_id="2faf500b" lines="30-44">
<description>
Remove each fracture marker line and push the surrounding object pixels one cell away from it.

Input structure
- A background grid with 8-connected foreground objects made from a body color and a rarer marker color.
- Each object has a non-square bbox; the marker pixels form one complete fracture line across the shorter bbox axis.

Rule
- Erase all marker pixels.
- For a wide object, each row has one marker; body pixels left of it shift one cell left and body pixels right of it shift one cell right.
- For a tall object, each column has one marker; body pixels above it shift one cell up and body pixels below it shift one cell down.

Key insight
- The marker line is a crack through the object, so the two sides separate by one pixel perpendicular to the crack.
</description>
</reference_puzzle>

<reference_puzzle index="10" source="train" puzzle_id="83eb0a57" lines="72-86">
<description>
Crop the largest object and overlay each smaller sprite where its shared marker pattern matches.

Input structure
- The background is the most common color, and the foreground has one largest destination object plus smaller source sprites.
- Each source sprite shares exactly one marker color with the destination object.

Rule
- Output the tight bbox crop of the largest object.
- Translate each source sprite so every connected marker component of the shared color exactly matches destination marker components.
- Paste source sprites in decreasing area order, so smaller sprites can sit on top of larger ones when they overlap.

Key insight
- The shared-color marker components are alignment keys; matching whole marker components distinguishes singleton slots from larger same-color marker blocks.
</description>
</reference_puzzle>

<reference_puzzle index="11" source="train" puzzle_id="7e2bad24" lines="91-106">
<description>
Extend the diagonal blue ray through the grid, reflecting it off red line-segment mirrors.

Input structure
- Black-background grid with one diagonal 8-connected blue ray seed.
- Red foreground objects, when present, are straight horizontal or vertical mirror segments.
- Other foreground bars are transparent to the ray and may be overwritten where the ray crosses them.

Rule
- Start from the ray seed endpoint whose outward diagonal extension remains inside the grid.
- Paint the ray continuously in that diagonal direction until it leaves the grid.
- If the next cell is a red mirror, do not paint it; reflect by flipping the row direction on a horizontal mirror or the column direction on a vertical mirror.

Key insight
- The non-red bars are decoys: only red bars reflect the diagonal path, while all other crossed cells become blue.
</description>
</reference_puzzle>

<reference_puzzle index="12" source="eval" puzzle_id="136b0064" lines="25-39">
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

<reference_puzzle index="13" source="eval" puzzle_id="4c3d4a41" lines="15-28">
<description>
Fit a grey bar into a grey frame by aligning projections with colored columns, then stack colors above.

Input structure:
- bg, a grey structural color forming two components: a large rectangular frame (slot) and a smaller bar with upward projections.
- Colored pixels inside the slot mark target columns.

Rule:
- Move the bar upward into the slot so its bottom edge sits one cell above the slot's interior bottom.
- Shift it horizontally so the bar's upward projections align column-wise with the colored pixels in the slot.
- Stack the colored pixels immediately above the bar projections in each column.

Key insight: The bar's projections act as keys that must align with the colored-column "locks" in the slot.
</description>
</reference_puzzle>

<reference_puzzle index="14" source="eval" puzzle_id="7c66cb00" lines="9-24">
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

<reference_puzzle index="15" source="train" puzzle_id="2601afb7" lines="30-45">
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

<reference_puzzle index="16" source="train" puzzle_id="5adee1b2" lines="8-23">
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

<reference_puzzle index="17" source="train" puzzle_id="da6e95e5" lines="70-81">
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

<reference_puzzle index="18" source="eval" puzzle_id="13e47133" lines="56-69">
<description>
Fill partitioned regions with concentric colored bands derived from marker pixels.

Input structure:
- Grid partitioned into regions by wall-colored lines.
- Each region has a dominant base color and a few isolated marker pixels of other colors at varying depths from the region boundary.

Rule:
- For each region, read marker colors ordered by their Chebyshev distance from the boundary to build a color palette.
- If no marker sits on the boundary itself, the base color leads the palette.
- Paint every cell with palette[distance mod palette_length], producing repeating concentric rectangular frames that radiate inward.

Key insight: Each marker's depth from the boundary defines its position in the cyclic palette; the pattern then tiles inward automatically.
</description>
</reference_puzzle>

<reference_puzzle index="19" source="train" puzzle_id="981add89" lines="10-29">
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

<reference_puzzle index="20" source="train" puzzle_id="e4888269" lines="67-88">
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

你的任务：
第一步：分析上述参考题中涉及的基础操作，思考这些操作之间有哪些非显而易见的串联方式。重点思考：
 - 一个操作的输出如何以意想不到的形式成为另一个操作的输入条件。
 - 信息可以通过哪些不直觉的方式编码在网格中。
 - 寻找已有操作之间新的因果关系和信息传递路径。

第二步：基于上述思考，开启头脑风暴，运用发散性思维，用中文创造至少5个全新的 ARC 谜题描述，若有较多合适的创新，也可以生成更多，不设上限。严格使用下面的 XML风格格式输出：

<new_puzzle>
- `description`：一句话概述题目的变换目标。
- `Input structure`：输入网格中有哪些对象、背景、标记、边界或组件。
- `Rule`：从输入到输出的确定性变换规则。
- `Key insight`：解题所需的核心洞见。
</new_puzzle>

注意事项：
1. 任务不应依赖大量先验知识或复杂计算，而应依赖对模式、结构与空间关系的直觉推理。
2. 若涉及计算，只允许使用简单的计数、排序与数量比较；不允许使用复杂的数学公式或运算。
3. 问题必须能从少量示例中推断出来，但不代表题目一定简单。设计时需从人类解题者的视觉角度思考：规则涉及的所有关键要素（对象、关系、属性）必须在网格上肉眼可见，不应依赖人眼无法直接观察到的隐含信息或需要推算的隐藏属性。规则可以很难发现，难到反复对比所有样例仍需要深度思考才能归纳出来，但一旦发现后必须能用来解释所有样例且产出唯一确定的输出。
4. 生成的问题描述旨在促进后续用于创建题目网格的代码生成。因此，对输入/输出网格的描述以及规则总结必须极其详细；语言风格应易于理解；对于特定对象（如箭头），必须在描述中明确其绘制方式。
5. 描述谜题时避免使用 0 或 1 这类颜色编号；应使用颜色名称，例如 “red”。可用颜色如下：Black、Blue、Red、Green、Yellow、Grey、Magenta、Orange、Azure、Maroon。
6. 五个新题的核心推理机制应明显不同——即从观察样例到归纳规则的认知路径不同，而不仅仅是场景、对象或颜色不同。
7.  题目难度应来自机制的隐蔽性而非步骤数。关键信息难以准确定位，需要反复对比尝试不同的角度才能发现真正的规则。


生成后对生成的题执行以下四项审查（审查内容不需要保存到文件）：
1. 原创性
  新题不可与上述20道参考题相同或者是类似，不可只是替换了颜色、方向、对象类型、维度等表层要素。判断标准：新题的核心推理路径（解题者从样例中归纳规则的认知过程）应与任何一道参考题不同。
2. 确定性
  规则完全确定，同一输入只能产生唯一输出，每一步操作都有明确定义。
3. 可实现性
  所有必要细节都已给出——对象如何生成、如何识别、如何变换、边界情况如何处理，读完描述后可以直接写代码。
4. ARC 合规性
  网格尺寸合理（≤30×30），使用颜色不超过 10 种，规则复杂度适中，人类能从 3-5个样例中发现规律。