你是一个设计 ARC-AGI（抽象与推理语料库，Abstraction and Reasoning Corpus）谜题的专家。你的任务是对下方提供的谜题逻辑与格式进行深入分析。通过参考示例谜题的推理与概念，你需要运用发散性思维进行创新扩展。请发挥创造力，构思全新且巧妙的推理谜题。
下面给出 20 个已有 ARC 题目的描述作为参考。每个描述通常包含：
- `description`：一句话概述题目的变换目标。
- `Input structure`：输入网格中有哪些对象、背景、标记、边界或组件。
- `Rule`：从输入到输出的确定性变换规则。
- `Key insight`：解题所需的核心洞见。

参考题目如下：
<reference_puzzle index="1" source="train" puzzle_id="30f42897" lines="51-66">
<description>
Repeat the single colored border segment around the rectangular border, leaving one segment-length gap between copies.

Input structure:
- The background is the dominant color.
- There is exactly one foreground color.
- All foreground pixels form one contiguous segment on the rectangular border, possibly crossing the cycle start.

Rule:
- Unwrap the grid border clockwise into a cyclic path.
- Let n be the length of the foreground segment.
- Copy the segment with the same color every 2*n border positions around the cycle.

Key insight:
- The foreground segment is not extended by rows or columns directly; it is a motif on the perimeter cycle, repeated with an equal-sized gap.
</description>
</reference_puzzle>

<reference_puzzle index="2" source="train" puzzle_id="edcc2ff0" lines="47-63">
<description>
Keep only the lower-panel colors named by the top markers and expand each marker into an object-count bar.

Input structure
A black instruction strip sits above a non-black rectangular scene panel.  The strip contains one-cell markers
in the first column, and each marker color is a candidate object color in the lower panel.  The lower panel has
a unique dominant background color and scattered 4-connected foreground objects.

Rule
In the lower panel, preserve objects whose colors appear as top markers and recolor every other foreground
object to the panel background.  In the instruction strip, replace each marker by a horizontal run from column 0
whose length is the number of 4-connected lower-panel objects of that marker color; erase markers with count 0.

Key insight
The top strip is a color filter plus a compact count summary: marker colors select which lower-panel objects
survive, and their bars report the number of surviving objects by color.
</description>
</reference_puzzle>

<reference_puzzle index="3" source="eval" puzzle_id="8f3a5a89" lines="22-36">
<description>
Paint an inner border around the region reachable from a corner marker pixel.

Input structure:
- bg = most common color; wall color (blue) forms barriers dividing the grid.
- A unique single pixel (magenta) sits on one corner — the start marker.
- Some wall-colored pixels are structural (border-touching, enclosing regions); others are decorations.

Rule:
- Flood-fill from the marker corner to find all reachable bg cells, blocked by structural walls.
- Erase wall-colored decorations hidden behind structural walls (not reachable from marker side).
- Paint orange (color 7) on every reachable bg cell adjacent to a wall or grid edge — forming a 1-cell inner border.

Key insight: Only border-connected wall components act as real barriers; interior wall fragments are decorations to be removed.
</description>
</reference_puzzle>

<reference_puzzle index="4" source="train" puzzle_id="57edb29d" lines="88-103">
<description>
Extract the single plain rectangle and redraw it with the aligned rectangles' border-relative marker pattern.

Input structure:
- The boundary color is the background/separator.
- Foreground objects are dense rectangular regions; exactly one rectangle has only its base color.
- Other rectangles contain their base color plus marker pixels.

Rule:
- Crop the only plain rectangle; this crop is the output canvas.
- Use marked rectangles whose row span overlaps the plain rectangle's row span.
- Transfer their marker pixels by preserving each marker run's distance from the nearest rectangle edges.

Key insight:
- The marked rectangles are examples of an edge-relative pattern, so the missing plain rectangle receives the same pattern resized to its own dimensions.
</description>
</reference_puzzle>

<reference_puzzle index="5" source="eval" puzzle_id="981571dc" lines="6-19">
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

<reference_puzzle index="6" source="eval" puzzle_id="4a21e3da" lines="8-20">
<description>
Cut a sprite along border-control rays and slide pieces to grid corners.

Input structure:
- bg, fg (main object, ~centered), ctrl (1-2 pixels on grid edges).

Rule:
- Draw rays inward from each ctrl pixel along its row/col, painting only bg cells up to the farthest fg pixel (intermittent).
- Rays partition the grid; fg on the cut line stays in place.
- Cut pieces in regions touching a control edge slide to that corner; others vanish.

Key insight: Only regions sharing a border edge with a control pixel keep their piece; the ray skips fg cells.
</description>
</reference_puzzle>

<reference_puzzle index="7" source="eval" puzzle_id="3e6067c3" lines="7-20">
<description>
Connect framed squares via colored bridges following a control-pixel sequence.

Input structure:
- bg = background. Border color forms equal-sized framed squares with unique inner fill colors.
- Control row below squares: colored pixels read left-to-right encode visitation order.

Rule:
- Start at the square matching first control color c0.
- For each consecutive pair (ci, ci+1), find the ci+1-colored square aligned in the same row or column band, preferring nearest.
- Draw a bridge colored ci between the pair, spanning the gap with inner-square thickness.

Key insight: Row/column-band alignment plus proximity resolves ambiguity when inner colors repeat.
</description>
</reference_puzzle>

<reference_puzzle index="8" source="train" puzzle_id="a416fc5b" lines="95-106">
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

<reference_puzzle index="9" source="eval" puzzle_id="8b7bacbf" lines="42-56">
<description>
Fill enclosed regions along a line seeded by unique border pixels.

Input structure:
- bg = most common color.
- Colored lines/drawings on the grid.
- Unique single-pixel colors on the grid border mark seed endpoints of each line.

Rule:
- Each border seed pixel connects (8-connected) to a line drawing.
- Find closed bg cavities whose boundary touches that line.
- Flood-fill each such cavity with the seed pixel's color.

Key insight: Only cavities reachable through the seed's own line get filled; the seed's unique color acts as a per-line fill label.
</description>
</reference_puzzle>

<reference_puzzle index="10" source="train" puzzle_id="ba1aa698" lines="41-55">
<description>
Extrapolate the moving sprite from a horizontal sequence of shared-border frames into the next standalone frame.

Input structure
- The grid is a row of equal-width framed panels sharing their vertical border columns.
- Each panel has the same fill color and one identical interior sprite at the same local columns.
- The sprite's top row changes by a constant amount from panel to panel.

Rule
- Build one output panel with the same height, frame color, fill color, and sprite.
- Continue the sprite's diagonal motion one more panel to the right.
- Because the output is a standalone panel, advance by the full panel width, not the shared-border spacing.

Key insight
- The panels are consecutive frames of motion; the output is the next frame after the sequence.
</description>
</reference_puzzle>

<reference_puzzle index="11" source="train" puzzle_id="470c91de" lines="48-65">
<description>
Move each teal-cornered filled rectangle one cell diagonally toward its teal corner.

Input structure:
- Dominant background surrounds several filled rectangular colored objects.
- Each object has exactly one teal control pixel replacing one corner cell of
  its rectangle.

Rule:
- Treat the teal pixel as the missing corner of its colored rectangle.
- Erase each original rectangle footprint, including the teal pixel.
- Repaint the completed rectangle shifted one cell diagonally in the direction
  of that teal corner.

Key insight:
- The control pixel's corner position gives both the missing cell to restore
  and the diagonal motion vector.
</description>
</reference_puzzle>

<reference_puzzle index="12" source="train" puzzle_id="6350f1f4" lines="62-81">
<description>
Decode a block-grid image by swapping each sprite-coded cell with a solid cell and each blank cell with the sprite.

Input structure
The grid is partitioned by black separator rows and columns into equal-size
square blocks. Each block is either a copy of one two-color sprite or a
uniform blank block of the sprite's minority color; grey pixels are erased
cells that act as wildcards.

Rule
Infer the repeated sprite from fully visible mixed-color blocks. Replace a
block matching that sprite, ignoring grey erasures, with a solid block of
the sprite's majority color. Replace a uniform blank block with the sprite.
Separator rows and columns, including grey erasures on them, become black.

Key insight
The small sprite is a block-level code for one macro-cell state, and the
uniform minority-color block is the other state; the output renders the
opposite code in each macro-cell while cleaning erased pixels.
</description>
</reference_puzzle>

<reference_puzzle index="13" source="train" puzzle_id="9473c6fb" lines="10-25">
<description>
Recolor the sparse foreground pixels in scan order along their longer spread, cycling red, teal, and grey.

Input structure:
- The dominant color is the background.
- Every foreground object is a single isolated 4-connected pixel.
- The foreground pixel set has a rectangular bbox whose row or column extent defines the scan direction.

Rule:
- Keep the background unchanged.
- If the foreground bbox is at least as tall as it is wide, order foreground pixels row-major; otherwise order them column-major.
- Recolor the ordered pixels by repeating red, teal, grey from the first pixel onward.

Key insight:
- The original foreground colors are only markers; the output color depends on each marker's ordinal position in the dominant scan direction.
</description>
</reference_puzzle>

<reference_puzzle index="14" source="train" puzzle_id="15660dd6" lines="33-54">
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

<reference_puzzle index="15" source="train" puzzle_id="c4d1a9ae" lines="28-48">
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

<reference_puzzle index="16" source="train" puzzle_id="1e5d6875" lines="26-45">
<description>
Paint a recolored diagonal copy from each L triomino source object.

Input structure:
- The background is the dominant color.
- The only foreground objects are 4-connected red and grey L triominoes,
  each occupying three pixels of a 2x2 box.

Rule:
- Grey L objects paint a yellow copy translated one diagonal step toward
  their missing 2x2 corner.
- Red L objects paint a green copy translated one diagonal step away from
  their missing 2x2 corner.
- Copies are clipped by the grid boundary; red-derived green copies are
  painted after grey-derived yellow copies where the projections collide.

Key insight:
- The missing corner of each 2x2 L object identifies its diagonal axis, and
  the source color determines which side of that axis receives the copy.
</description>
</reference_puzzle>

<reference_puzzle index="17" source="train" puzzle_id="984d8a3e" lines="5-21">
<description>
Shift each row's dominant middle-color run left until its right edge aligns with the leftmost original right edge, moving displaced left-side pixels after the run.

Input structure:
- The grid has three colors.
- The most common color forms one contiguous horizontal run in every row.
- Pixels before those runs share one side color, and pixels after those runs share the other side color.

Rule:
- Find the minimum right edge column among the middle-color runs.
- In each row, keep the run width and the trailing right-side color count.
- Place the middle-color run so its right edge is at that minimum column, unless the run is too wide, in which case start it at column 0.
- Fill any remaining non-trailing cells with the left-side color.

Key insight:
The varying row prefixes are movable padding; the stable feature is the leftmost right boundary of the middle-color shape.
</description>
</reference_puzzle>

<reference_puzzle index="18" source="train" puzzle_id="6bcdb01e" lines="10-24">
<description>
Extend the edge-touching green start bar through the corridor, turning at each teal blocker until the path leaves the grid.

Input structure:
- The background is the most common color.
- Exactly two foreground colors are present: a 2-pixel 4-connected start bar and many blocker pixels.
- Exactly one pixel of the start bar lies on the grid border.

Rule:
- Start from the border pixel of the bar and move through the other bar pixel into the grid.
- Paint every traversed non-blocker pixel with the start color.
- When the next cell is a blocker, turn to the unique perpendicular non-blocker neighbor; when the next cell is outside the grid, stop.

Key insight: The blocker pixels define a one-cell-wide corridor, so every interior collision has exactly one valid 90-degree turn.
</description>
</reference_puzzle>

<reference_puzzle index="19" source="train" puzzle_id="25c199f5" lines="30-44">
<description>
Pack the non-background rows from three separated panels into one panel, scanning panels right-to-left.

Input structure:
- A rectangular grid contains three equal-size panels separated by two full-height divider columns.
- The background is the dominant color; divider pixels use a color that appears only in the divider columns.

Rule:
- Read panels from right to left, and within each panel read rows from top to bottom.
- Keep each row that contains any non-background pixel, discarding all-background rows.
- Place the kept rows at the bottom of a single panel-shaped output, padding above with background rows.

Key insight:
- The task is row packing: the divider columns mark independent panels, and only the occupied rows are moved.
</description>
</reference_puzzle>

<reference_puzzle index="20" source="eval" puzzle_id="89565ca0" lines="120-136">
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