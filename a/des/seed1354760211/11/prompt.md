你是一个设计 ARC-AGI（抽象与推理语料库，Abstraction and Reasoning Corpus）谜题的专家。你的任务是对下方提供的谜题逻辑与格式进行深入分析。通过参考示例谜题的推理与概念，你需要运用发散性思维进行创新扩展。请发挥创造力，构思全新且巧妙的推理谜题。
下面给出 20 个已有 ARC 题目的描述作为参考。每个描述通常包含：
- `description`：一句话概述题目的变换目标。
- `Input structure`：输入网格中有哪些对象、背景、标记、边界或组件。
- `Rule`：从输入到输出的确定性变换规则。
- `Key insight`：解题所需的核心洞见。

参考题目如下：
<reference_puzzle index="1" source="train" puzzle_id="78e78cff" lines="71-91">
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

<reference_puzzle index="3" source="train" puzzle_id="c803e39c" lines="31-40">
<description>
Expand the second mask into a block grid by stamping the first mask with the two payload colors.

Input structure
- A background grid has four panels separated by three full-height separator columns.
- The first two panels contain same-size binary masks; the last two contain solid same-size color squares.
Rule
- Use the second mask as an outer layout. For each foreground cell in that outer mask, place a tile equal to the first mask, coloring its foreground with the first solid-square color and its background with the second solid-square color; outer background cells become all second solid-square color.
Key insight
- The output is the Kronecker product of the two binary masks, with the two right-hand panel colors providing the palette.
</description>
</reference_puzzle>

<reference_puzzle index="4" source="train" puzzle_id="963c33f8" lines="86-108">
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

<reference_puzzle index="5" source="eval" puzzle_id="a6f40cea" lines="74-88">
<description>
Complete partial paths inside a framed rectangle by projecting external color traces inward.

Input structure:
- bg, a 1px rectangular frame (border color) with uniform interior fill.
- Colored components outside the frame touching its edges.

Rule:
- Output = the frame's interior, filled with the fill color.
- Map each outside component's frame-touch points into interior coordinates.
- Monochrome components become solid Manhattan stripes between mapped endpoints.
- Two-color components become alternating-color Manhattan paths.

Key insight: Endpoint colors of two-color traces anchor the alternating phase deterministically.
</description>
</reference_puzzle>

<reference_puzzle index="6" source="eval" puzzle_id="271d71e2" lines="9-22">
<description>
Move bordered containers toward same-colored destination markers, charging gray cells to orange.

Input structure:
- bg = 6 (magenta). Black-bordered (0) rectangles with gray (5) / orange (7) interiors.
- A head sprite (9) sits adjacent to one container edge; a matching destination sprite lies along the movement axis.

Rule:
- Each container translates toward its destination along the head-to-dest axis.
- Steps = min(head-to-dest distance, number of gray interior cells).
- Gray cells become orange from the head-facing side inward, one per step.

Key insight: Gray cell count caps movement — a container stops short if it runs out of gray cells to charge.
</description>
</reference_puzzle>

<reference_puzzle index="7" source="train" puzzle_id="d6e50e54" lines="8-25">
<description>
Move each maroon marker toward the blue rectangle and recolor the rectangle red.

Input structure
A background canvas contains one solid blue rectangular object and several isolated
maroon marker pixels. Each marker is horizontally or vertically aligned with the
rectangle and lies outside it.

Rule
Start with a background canvas, repaint the blue rectangle red, and move each
maroon marker straight toward the rectangle along its row or column. Exactly one
marker is strictly closer to the rectangle than all others and enters the
rectangle boundary; every other marker lands in the adjacent exterior cell just
outside its aligned side.

Key insight
The single entering marker is the unique global nearest marker to the rectangle,
not the nearest marker on each of the four sides.
</description>
</reference_puzzle>

<reference_puzzle index="8" source="train" puzzle_id="b7955b3c" lines="386-403">
<description>
Replace gray plus-shaped damage masks with the hidden colors of the rectilinear mosaic.

Input structure:
- Color 8 appears only as isolated 3x3 plus masks.
- The non-mask pixels are visible parts of a clean grid made from horizontal
  and vertical color regions.

Rule:
- Treat every color-8 pixel as unknown damage and leave all other pixels fixed.
- First copy unknown pixels from exact matching parallel rows or columns.
- Then fill any remaining mask cells from opposite-side continuations and the
  better-supported local straight edge.

Key insight:
- The gray pluses are occluders, not objects; the underlying colors are
  determined by repeated line templates and nearby axis-aligned boundaries.
</description>
</reference_puzzle>

<reference_puzzle index="9" source="train" puzzle_id="aaef0977" lines="25-41">
<description>
Replace an orange canvas with concentric Manhattan-distance color rings centered on the lone marker pixel.

Input structure:
- The grid is an orange background with exactly one non-orange marker pixel.
- The marker color is one of the colors in the fixed non-orange ring cycle.

Rule:
- The marker location is distance 0 and keeps the marker color.
- Each step of Manhattan distance away from the marker advances one position
  through the cycle green, yellow, black, grey, red, teal, maroon, pink, blue.
- Fill every pixel of the original grid by this distance-indexed cycle.

Key insight: The output is not copied from existing objects; it is the
distance field around the single marker, colored by a cyclic palette that
excludes the orange background.
</description>
</reference_puzzle>

<reference_puzzle index="10" source="train" puzzle_id="1be83260" lines="41-58">
<description>
Decode the colored slot map inside the marked motif and rebuild the motif lattice with those colors.

Input structure:
- A blank background surrounds a rectangular lattice of identical motifs.
- The motif rows and columns are separated by one background line.
- One motif has colored marker pixels in its internal odd-row/odd-column slots.

Rule:
- The motif size is 2R+1 by 2C+1, where R by C is the motif lattice size.
- Read the marked motif's odd internal slots as the color map for the R by C output motifs.
- Slots that still contain the repeated motif color map to that repeated motif color.
- Rebuild the compact lattice on a canvas filled with the repeated motif color,
  painting each motif mask with its decoded color.

Key insight:
- The holes and filled odd slots inside one motif are a miniature legend for recoloring the entire repeated motif lattice.
</description>
</reference_puzzle>

<reference_puzzle index="11" source="eval" puzzle_id="c7f57c3e" lines="22-35">
<description>
Swap every sprite to the opposite prototype while preserving scale and position.

Input structure:
- bg = most common color. Two distinct small sprite patterns (prototypes) appear at various scales.
- Each sprite instance is a scaled copy of one of the two prototypes, placed on the grid.

Rule:
- Identify the two prototype patterns from the smallest-area sprites.
- For each sprite, determine which prototype it matches and its integer scale factor.
- Replace it with the other prototype at the same scale, aligned by an internal anchor region (color-1 pixels).

Key insight: Every sprite toggles to the alternate prototype; the anchor sub-region inside each sprite pins the alignment so swapped sprites land in the correct position regardless of scale.
</description>
</reference_puzzle>

<reference_puzzle index="12" source="train" puzzle_id="5034a0b5" lines="16-32">
<description>
Shift each interior pixel whose color matches a frame side one cell toward that matching side.

Input structure
The grid has a one-cell-thick rectangular frame: the four corners are the
background color, and each side between corners is a solid, distinct side
color. The interior is mostly background with sparse foreground pixels.

Rule
Interior pixels with the top-side color move one row up, bottom-side color
move one row down, left-side color move one column left, and right-side color
move one column right. A pixel already adjacent to its matching side stays in
place; colors that do not match a frame side and the frame itself are unchanged.

Key insight
The colored frame sides act as direction labels for matching interior pixels.
</description>
</reference_puzzle>

<reference_puzzle index="13" source="train" puzzle_id="ad173014" lines="97-111">
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

<reference_puzzle index="14" source="train" puzzle_id="305b1341" lines="22-37">
<description>
Expand each sparse color object into a filled rectangular region using its legend-paired fill color.

Input structure
A background-color grid has a two-column legend in columns 0 and 1, starting at row 0: each row gives
a source color in the left column and that source color's fill color in the right column.  Outside the
legend, the foreground pixels are sparse dot objects whose colors are exactly the legend source colors.

Rule
Clear the legend.  For each source color, take the bbox of all of its non-legend dots, expand that bbox
by one pixel in every direction, and fill background cells inside the expanded rectangle with the paired
legend fill color while preserving every original non-legend dot.

Key insight
The legend order is also the layer order: earlier legend rows are in front, so fill rectangles are painted
from the last legend row back to the first and higher-priority fills overwrite lower-priority fills.
</description>
</reference_puzzle>

<reference_puzzle index="15" source="train" puzzle_id="30f42897" lines="51-66">
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

<reference_puzzle index="16" source="train" puzzle_id="b25e450b" lines="28-43">
<description>
Slide each black border rectangle to the opposite border, recoloring its swept stripe orange and leaving the rectangle black at the far edge.

Input structure
- The grid contains black rectangular 4-connected masks, each touching exactly one outer border.
- Grey cells form incidental pattern content; orange is the trail color used when a black mask slides.

Rule
- Treat all black rectangles as moving simultaneously.
- A rectangle touching the top or bottom edge moves vertically to the opposite edge in the same columns.
- A rectangle touching the left or right edge moves horizontally to the opposite edge in the same rows.
- Recolor the complete swept row or column stripe to orange, then stamp every destination rectangle in black.

Key insight
- Edge contact gives the slide direction; the output is the set of opposite-edge black rectangles plus their orange trails.
</description>
</reference_puzzle>

<reference_puzzle index="17" source="train" puzzle_id="fbf15a0b" lines="49-69">
<description>
Crop out the grid half containing the two aligned marker pixels, then erase the markers to background.

Input structure
- The grid has a dominant background color and one non-background marker color
  that appears as exactly two cells.
- The marker cells are aligned either in one row or one column and both lie
  in the same half along that direction.
- Other non-background pixels are content inside one or both halves.

Rule
- If the markers share a row, keep the left or right half whose columns
  contain them.
- If the markers share a column, keep the top or bottom half whose rows
  contain them.
- In the kept half, recolor the marker pixels to the background color.

Key insight
- The two marker pixels are not part of the picture; their alignment and
  half-position identify which half of the doubled canvas is the answer.
</description>
</reference_puzzle>

<reference_puzzle index="18" source="eval" puzzle_id="981571dc" lines="6-19">
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

<reference_puzzle index="19" source="eval" puzzle_id="b0039139" lines="29-44">
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

<reference_puzzle index="20" source="train" puzzle_id="7ec998c9" lines="14-28">
<description>
Draw a blue folded corner-to-corner route through the lone marker's column.

Input structure:
- A rectangular grid filled with one background color.
- Exactly one non-background marker pixel; blue is absent from the input.

Rule:
- Preserve the marker and repaint background pixels in its column blue from top to bottom.
- If the marker is in the unique center column, connect that column to the top-right and bottom-left corners.
- Otherwise, connect the column to the top-left and bottom-right corners.

Key insight:
The marker's column is the route spine; its central vs off-center placement selects which opposite corner pair gets the horizontal edge segments.
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