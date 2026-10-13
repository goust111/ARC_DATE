你是一个设计 ARC-AGI（抽象与推理语料库，Abstraction and Reasoning Corpus）谜题的专家。你的任务是对下方提供的谜题逻辑与格式进行深入分析。通过参考示例谜题的推理与概念，你需要运用发散性思维进行创新扩展。请发挥创造力，构思全新且巧妙的推理谜题。
下面给出 20 个已有 ARC 题目的描述作为参考。每个描述通常包含：
- `description`：一句话概述题目的变换目标。
- `Input structure`：输入网格中有哪些对象、背景、标记、边界或组件。
- `Rule`：从输入到输出的确定性变换规则。
- `Key insight`：解题所需的核心洞见。

参考题目如下：
<reference_puzzle index="1" source="eval" puzzle_id="8b9c3697" lines="9-23">
<description>
Insert red components into interior sockets of large structures, leaving a black trail.

Input structure:
- bg = most common. Large non-bg, non-red structures with interior cavities.
- Red components outside structures.

Rule:
- Interior sockets: bg cells inside a structure's bbox, adjacent to structure pixels.
- For each structure, find the direction where rays from sockets hit a red component.
- Slide that component into the socket; paint traversed cells black.
- Unmatched reds vanish.

Key insight: Docking direction is inferred by ray-casting from interior sockets to the nearest red component.
</description>
</reference_puzzle>

<reference_puzzle index="2" source="train" puzzle_id="84ba50d3" lines="36-54">
<description>
Drop the blue sprites downward relative to the red divider, with vertical sprites passing through to the bottom.

Input structure:
- The background is the dominant color.
- One non-background color forms a single full-width horizontal divider row.
- The other non-background color forms 4-connected sprites above the divider.

Rule:
- Clear the original sprite pixels.
- A one-column sprite falls to the bottom of its column and deletes the divider
  pixel it crossed.
- Any wider sprite falls until its unique widest horizontal row is immediately
  above the divider row; if the moved sprite overlaps the divider, it covers it.

Key insight:
The red row is not a generic floor: only one-column sprites pass through to the
bottom, while wider sprites are registered by their widest horizontal segment.
</description>
</reference_puzzle>

<reference_puzzle index="3" source="train" puzzle_id="bd5af378" lines="25-42">
<description>
Rewrite a corner L block into two nested L borders and a diagonal trace.

Input structure
A rectangular grid has exactly two colors: one color forms a one-cell-thick
L on two adjacent outer edges, and the other color fills the remaining
rectangle.

Rule
Normalize the L to the top-left corner. Paint the outer top-left L with the
fill color, the next inner L with the original border color, and the main
diagonal of the remaining inner rectangle with the fill color. All other
cells become color 8, then the original orientation is restored.

Key insight
The corner of the input L determines the orientation; after flipping to a
canonical top-left L, the output is the same nested-layer pattern for every
example.
</description>
</reference_puzzle>

<reference_puzzle index="4" source="train" puzzle_id="db615bd4" lines="111-130">
<description>
Thicken the sparse frame and stamp the external sparse sprites into it in positional order.

Input structure
The grid has a dominant background color, a regular lattice color, one large
sparse rectangular frame, and one or more external monochrome sprites whose
pixels also sit on a two-step lattice.

Rule
Use the largest sparse rectangular perimeter as the frame. Clear that frame
region and the external sprite pixels to background, draw a solid one-pixel
border in the frame color, solidify each sparse sprite into its contiguous
two-step mask, and place the sprites inside the frame with equal gaps.
Wide frames receive sprites left-to-right by source position; tall frames
receive them top-to-bottom by source position.

Key insight
The frame supplies ordered slots: the sprite-to-slot assignment is by visible
source order, not by numeric color or by matching individual cells.
</description>
</reference_puzzle>

<reference_puzzle index="5" source="train" puzzle_id="8e301a54" lines="8-23">
<description>
Move every foreground object straight downward by a number of rows equal to its own pixel count.

Input structure:
- The grid has one dominant background color.
- The foreground is made of separated 4-connected monochromatic objects.
- Objects may move partly below the bottom edge of the grid.

Rule:
- Clear the canvas to the background.
- For each foreground object, compute its area in pixels.
- Redraw the object at the same columns and that many rows lower, clipping pixels below the grid.

Key insight:
- The downward displacement is the object's own area; color and shape only identify which pixels move together.
</description>
</reference_puzzle>

<reference_puzzle index="6" source="train" puzzle_id="b2bc3ffd" lines="8-23">
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

<reference_puzzle index="7" source="eval" puzzle_id="5961cc34" lines="8-21">
<description>
Build a red ray network by activating blue blobs via directional green ports.

Input structure:
- bg = most common color. Large blue blobs with adjacent green pixels marking port directions.
- Start marker: yellow pixel atop a short red segment at grid bottom.

Rule:
- Extend a ray upward from start; on hitting a blue blob, recolor it red.
- Activated blobs emit thick rays through green ports until boundary or another blob.
- Cascade until no new blobs are reached.

Key insight: Green pixels encode ray direction and width; the network activates as a chain reaction from one start point.
</description>
</reference_puzzle>

<reference_puzzle index="8" source="train" puzzle_id="e3fe1151" lines="36-55">
<description>
Fill the one blank in each corner region so all four regions have the same color multiset.

Input structure:
- The dominant background color forms one solid separator row and one solid
  separator column.
- The separator cross splits the grid into four equal rectangular corner regions.
- Each corner region contains exactly one background-colored blank pixel.

Rule:
- Count the non-background colors in each corner region.
- The completed region multiset is defined by the maximum observed count of
  each color across the four regions.
- Replace each region's blank pixel with the single color needed to match that
  completed multiset.

Key insight:
- The separator color inside a corner region is a missing value marker; comparing
  the four regions reveals the shared multiset without using numeric color order.
</description>
</reference_puzzle>

<reference_puzzle index="9" source="train" puzzle_id="22806e14" lines="23-37">
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

<reference_puzzle index="10" source="train" puzzle_id="bc93ec48" lines="48-63">
<description>
Copy each foreground object anchored in a true grid corner clockwise into the next corner.

Input structure
- The dominant color is the background.
- Foreground objects are monochromatic 4-connected components.
- Relevant objects are exactly those whose bounding box touches two adjacent borders at one grid corner.

Rule
- Start from the input unchanged.
- For each corner object, copy its pixel mask and color to the next corner clockwise: top-left to top-right, top-right to bottom-right, bottom-right to bottom-left, and bottom-left to top-left.
- Preserve each object's orientation by aligning its bounding box to the destination corner; stamps may overwrite existing pixels at the destination, but source pixels are not cleared.

Key insight
- Interior and side-only objects are distractors; only components actually anchored to a corner participate in the clockwise perimeter copy.
</description>
</reference_puzzle>

<reference_puzzle index="11" source="train" puzzle_id="22425bda" lines="58-76">
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

<reference_puzzle index="12" source="train" puzzle_id="78e78cff" lines="71-91">
<description>
Recolor the row-wise passage indicated by wall pixels with the singleton marker color.

Input structure:
- A dominant background contains one singleton marker pixel and one wall color.
- Wall pixels lie on several rows and form left/right limits around the marker column.
- Empty rows between the top and bottom wall rows are open horizontal passages.

Rule:
- On a wall row, paint background pixels between the nearest wall strictly
  left of the marker column and the nearest wall at or right of it; if only
  one side is present, paint from that wall to the opposite grid edge.
- On empty rows between the top and bottom wall rows, paint the whole row.
- On empty rows outside the wall-row band, reuse the nearest end wall-row interval.
- Preserve wall pixels and the original marker pixel.

Key insight:
- The wall color defines scanline limits relative to the marker column; the
  marker color fills those implied horizontal spans rather than flood-filling
  through connected background.
</description>
</reference_puzzle>

<reference_puzzle index="13" source="eval" puzzle_id="13e47133" lines="56-69">
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

<reference_puzzle index="14" source="eval" puzzle_id="3a25b0d8" lines="9-22">
<description>
Fill holes in a target object with colors from a source object's subsprites, then tight-crop.

Input structure:
- bg = background. Two similar 8-connected objects share an outline color.
- Source has colored subsprites in cavities; target has empty cavities.

Rule:
- Find the two 8-connected objects.
- Target = object with more hole area. Align subsprites to holes (rotation/flip if needed).
- Fill each hole with its matched subsprite color; tight-crop.

Key insight: Hole centers in the target are matched to source subsprite centers under the best rotation/flip transform.
</description>
</reference_puzzle>

<reference_puzzle index="15" source="train" puzzle_id="c3fa4749" lines="79-96">
<description>
Recolor each side-marked multicolor object embedded in a monochrome panel to its marker color.

Input structure:
- Same-color 4-connected components form monochrome panel regions.
- The bounding box of a panel may contain 4-connected non-panel-color objects.
- A target object touches exactly one side of its panel bbox, and all of its
  pixels on that side have one marker color.

Rule:
- For every target object, fill any panel-colored holes enclosed by the object.
- Recolor the object and filled holes to the side marker color.
- Leave all other pixels unchanged.

Key insight:
- A multicolor object's single contact side identifies both which object should
  be normalized and which color should replace it.
</description>
</reference_puzzle>

<reference_puzzle index="16" source="train" puzzle_id="a416fc5b" lines="95-106">
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

<reference_puzzle index="17" source="train" puzzle_id="fd02da9e" lines="26-41">
<description>
Replace each colored corner marker with its corner-specific inward stencil in the same color.

Input structure
- The grid is 8x8 with a single dominant background color.
- Foreground pixels are marker pixels, each located at one of the four corners.
- A marker's color is carried to the stencil selected by its corner position.

Rule
- Erase each corner marker back to the background.
- Top corner markers become an inward 2x2 block one cell away from the top and side edges.
- Bottom corner markers become the mirrored three-pixel inward bent stencil shown by the bottom-corner examples.

Key insight
- The marker position selects the fixed stencil; the marker color only supplies the paint color, so multiple corners apply independently.
</description>
</reference_puzzle>

<reference_puzzle index="18" source="train" puzzle_id="a6953f00" lines="45-63">
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

<reference_puzzle index="19" source="train" puzzle_id="99caaf76" lines="56-75">
<description>
Move each marked cross object away from its marker to the grid edge while rotating the colored cross 180 degrees.

Input structure:
- The background is the dominant color.
- Each 8-connected foreground object has nine pixels: a 4-pixel marker-color
  sprite attached to a 5-pixel non-marker colored cross.
- Each object bbox is either 3x6 or 6x3, so the marker sprite is on exactly
  one side of the cross.

Rule:
- Move the whole object straight away from its marker side until the object
  bbox touches the corresponding grid edge.
- Keep the marker sprite unchanged within the object.
- Rotate the 3x3 colored cross block by 180 degrees before placing it.

Key insight:
The marker sprite is a directional tail; the cross is the leading payload,
and only that payload turns around during the slide to the edge.
</description>
</reference_puzzle>

<reference_puzzle index="20" source="train" puzzle_id="fc4aaf52" lines="17-34">
<description>
Split the two-color object horizontally, swap its colors, and slide the upper half to the right.

Input structure:
- The grid has one background color and exactly two foreground colors.
- All foreground pixels form one 4-connected object with an even-height bbox.
- The object has foreground pixels on both rows adjacent to its horizontal midpoint.

Rule:
- Split the object into upper and lower halves at the midpoint of its bbox.
- Swap the two foreground colors in both halves.
- Keep the lower half in place and shift the upper half right until its lower-left
  boundary pixel is diagonally adjacent to the lower half's upper-right boundary pixel.

Key insight:
- The object is not copied or reflected; it is broken at the horizontal waist,
  recolored by exchanging the two visible object colors, then rejoined diagonally.
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