你是一个设计 ARC-AGI（抽象与推理语料库，Abstraction and Reasoning Corpus）谜题的专家。你的任务是对下方提供的谜题逻辑与格式进行深入分析。通过参考示例谜题的推理与概念，你需要运用发散性思维进行创新扩展。请发挥创造力，构思全新且巧妙的推理谜题。
下面给出 20 个已有 ARC 题目的描述作为参考。每个描述通常包含：
- `description`：一句话概述题目的变换目标。
- `Input structure`：输入网格中有哪些对象、背景、标记、边界或组件。
- `Rule`：从输入到输出的确定性变换规则。
- `Key insight`：解题所需的核心洞见。

参考题目如下：
<reference_puzzle index="1" source="train" puzzle_id="34cfa167" lines="94-117">
<description>
Complete a four-corner framed rectangle from one decorated anchor rectangle and its diagonal mate.

Input structure:
- The background is the dominant color.
- Two same-color solid rectangular anchors of the same size sit on opposite
  diagonal corners, with the first anchor above-left of the second.
- The upper-left anchor has two one-cell-thick vertical control bars to its
  right and two one-cell-thick horizontal control bars below it.

Rule:
- Copy the anchor rectangle into the two missing corners of the implied
  larger rectangle.
- Repeat the right-side control bars horizontally between the left and right
  anchors, using their spacing as the period, and draw the outer top and
  bottom borders in the first right-control color.
- Repeat the below-side control bars vertically between the top and bottom
  anchors, using the same period, and draw the outer left and right borders
  in the first below-control color.

Key insight: The small control bars are the first two ticks of two periodic
frame patterns; the diagonal anchor supplies the opposite corner and the
missing two anchor copies.
</description>
</reference_puzzle>

<reference_puzzle index="2" source="train" puzzle_id="963c33f8" lines="86-108">
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

<reference_puzzle index="3" source="train" puzzle_id="db7260a4" lines="37-56">
<description>
Drop the singleton marker into the matching red U-shaped container, or make a bottom-row puddle if no closed container is hit.

Input structure
- The grid has a dominant background, one singleton marker pixel on the top row,
  and one other foreground color forming vertical bars and bottom segments.
- A closed container is a U shape: two vertical side bars with a continuous
  bottom segment between them and an empty rectangular interior above the bottom.

Rule
- Remove the top marker from its original cell.
- If exactly one closed U-shaped container has the marker column within its
  horizontal footprint, fill that container's empty interior with the marker color.
- If no closed container is under the marker column, fill the whole bottom row
  with the marker color.

Key insight
- The marker does not identify a color by value; it identifies a vertical drop
  column, and only a structurally closed U shape can catch and contain the fill.
</description>
</reference_puzzle>

<reference_puzzle index="4" source="eval" puzzle_id="2ba387bc" lines="6-21">
<description>
Sort scattered 4×4 squares into two columns by type: frames left, solids right.

Input structure:
- bg = most common color. Multiple 4×4 objects on bg.
- Frames: colored 1px border, bg-filled 2×2 interior.
- Solids: entirely filled with one non-bg color.

Rule:
- Classify each 4×4 object as frame or solid by its interior.
- Sort each group top-to-bottom by input row.
- Stack frames in left column (cols 0–3), solids in right (cols 4–7).
- Output is 8 wide; height = 4 × max(#frames, #solids), bg-padded.

Key insight: Frame vs solid is determined solely by whether the 2×2 interior matches bg.
</description>
</reference_puzzle>

<reference_puzzle index="5" source="eval" puzzle_id="89565ca0" lines="120-136">
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

<reference_puzzle index="6" source="train" puzzle_id="c9680e90" lines="52-67">
<description>
Move each mirrored bottom source pixel to the far end of its trail and redraw the top half as its reflection.

Input structure
- A solid non-background row separates equal-height top and bottom halves.
- The top half has one marker color on the background.
- In the bottom half, exactly one color has positions that mirror the top markers across the separator, and the other color forms trails touching those source pixels.

Rule
- Identify the bottom source color by reflecting each bottom color into the top half and matching the top marker set.
- For every bottom source pixel, follow its unique 4-connected trail of the other bottom color to the trail endpoint.
- Clear the old top markers, bottom sources, and trails; place the source color at each endpoint and place the top marker color at the reflected endpoint.

Key insight
- The top half is only a mirror display of the bottom moving sources; the actual motion is encoded by the trail attached to each bottom source pixel.
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

<reference_puzzle index="8" source="train" puzzle_id="22425bda" lines="58-76">
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

<reference_puzzle index="9" source="train" puzzle_id="fd02da9e" lines="26-41">
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

<reference_puzzle index="10" source="train" puzzle_id="753ea09b" lines="60-79">
<description>
Fill the small background compartments bounded by the long contour color.

Input structure:
- The grid has one dominant background color and two foreground colors.
- The more frequent foreground color forms long 8-connected contour fragments.
- The other foreground color is a fixed distractor/obstacle color that partitions
  the background but is never recolored.

Rule:
- Split the background into 4-connected regions using all foreground pixels as
  barriers.
- Keep the two largest background regions as the outside of the contour drawing.
- Recolor every other background region that touches a long contour component
  to the contour color; leave all foreground pixels unchanged.

Key insight:
The apparent area fill is exactly a component-level fill of bounded background
compartments, not a row-wise interpolation between individual contour pixels.
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

<reference_puzzle index="12" source="eval" puzzle_id="13e47133" lines="56-69">
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

<reference_puzzle index="13" source="eval" puzzle_id="b99e7126" lines="34-48">
<description>
Scale a minority cell's inner pattern up to the cell-grid level, completing a fractal overlay.

Input structure:
- Grid divided into NxN cells by separator lines of a single color.
- Most cells share an identical interior pattern (majority); a few have a distinct pattern (minority).

Rule:
- Extract the inner pixel pattern from the minority cell.
- Treat each pixel as a cell-sized stamp: active pixels → minority pattern, inactive → majority pattern.
- Position this scaled-up pattern on the cell grid so existing minority cells align with active pixels.
- Fill in all remaining active positions with the minority pattern.

Key insight: The minority cell's interior is a miniature blueprint of where minority cells should appear at the grid scale — fractal self-similarity.
</description>
</reference_puzzle>

<reference_puzzle index="14" source="eval" puzzle_id="800d221b" lines="7-22">
<description>
Recolor a path-network by adjacent seed regions.

Input structure:
- bg = most common color.
- Path color P forms a thin orthogonal network with a solid square hub (NxN).
- Seed colors appear in colored regions (non-bg, non-P) adjacent to path branches.

Rule:
- Infer P as the non-bg color whose largest component is most network-like.
- Find the largest solid k×k all-P square (the hub); exclude it from recoloring.
- Recolor each path branch to the seed-region color it touches most.
- Recolor the hub center pixel to the seed whose branches touch the hub most.

Key insight: The hub stays uncolored; branches inherit color from adjacent seed regions by adjacency count.
</description>
</reference_puzzle>

<reference_puzzle index="15" source="train" puzzle_id="9b5080bb" lines="161-172">
<description>
Recolor the small defect bars on each shared-color object with the opposite panel color.

Input structure
- The grid contains two large panel colors and one shared object color.
- Each shared-color object is a mostly solid rectangle inside exactly one panel.
Rule
- Find the near-solid rectangular core of each object.
- Missing core pixels and one-cell protrusions define two-pixel bars perpendicular to the nearest core edge.
- Recolor those bars with the other panel color; leave all other pixels unchanged.
Key insight
- The colors to add are not tied to numeric ids: each object's surrounding panel identifies the opposite panel color.
</description>
</reference_puzzle>

<reference_puzzle index="16" source="train" puzzle_id="42f14c03" lines="75-83">
<description>
Crop the foreground object whose missing cells match marker shapes, and fill those missing cells with the marker colors.

Input structure
A dominant background surrounds several monochromatic foreground objects. Exactly one object has background gaps inside its bounding box whose component shapes are matched by separate marker objects.
Rule
Find that object, crop its bounding box, and replace each background gap component with the color of the separate marker component or components having the same normalized shape.
Key insight
The marker objects do not mark absolute positions; their shapes identify which gaps in the selected object should be colored, while objects with unmatched gaps are distractors.
</description>
</reference_puzzle>

<reference_puzzle index="17" source="train" puzzle_id="33067df9" lines="32-48">
<description>
Expand the sparse dot lattice into a 26x26 drawing with thick bars between adjacent matching dots.

Input structure
- The background is the dominant color.
- Foreground pixels lie on logical rows and columns separated by background spacer rows/columns.
- A logical row or column is present when it contains at least one foreground pixel.

Rule
- Expand every background spacer row/column to thickness 2.
- Expand the logical rows/columns equally so the full output is 26x26.
- Paint each foreground logical pixel as a solid rectangle of its color.
- Connect adjacent same-color logical pixels horizontally through the intervening spacer.
- Connect adjacent same-color logical pixels vertically only when neither endpoint is part of a horizontal same-color run.

Key insight
- The input is a sampled line drawing: horizontal same-color runs define the bars first, and only unclaimed same-color vertical neighbors are joined.
</description>
</reference_puzzle>

<reference_puzzle index="18" source="train" puzzle_id="984d8a3e" lines="5-21">
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

<reference_puzzle index="19" source="train" puzzle_id="8a6d367c" lines="52-67">
<description>
Complete the framed area by expanding the unique upper template that contains the existing seed pixels.

Input structure:
- A teal background contains small colored template masks above a larger rectangular frame.
- The frame encloses one fill color arranged as a partial block-scaled copy of one template.
- The frame interior dimensions are an integer square scale of the matching template bbox.

Rule:
- Crop to the rectangular frame.
- Identify the unique upper template whose block-scaled mask covers every existing fill pixel inside the frame.
- Reset the frame interior to background and paint the whole scaled mask with the fill color.

Key insight:
The lower colored pixels are seed cells that reveal which upper template should be completed inside the frame.
</description>
</reference_puzzle>

<reference_puzzle index="20" source="eval" puzzle_id="fc7cae8d" lines="9-21">
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