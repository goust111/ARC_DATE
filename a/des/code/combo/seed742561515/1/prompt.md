你是一个 ARC-AGI 谜题设计专家。下面的 20 道参考题是难度和机制质量的标杆。仔细阅读每道题的**解题逻辑**（解题者从样例归纳规则的认知过程），而非表面场景。你的新题应在机制深度和隐蔽性上与参考题持平或者实现超越。

每个参考题描述包含：
- `description`：一句话概述题目的变换目标。
- `Input structure`：输入网格中有哪些对象、背景、标记、边界或组件。
- `Rule`：从输入到输出的确定性变换规则。
- `Key insight`：解题所需的核心洞见。

参考题目如下：
<reference_puzzle index="1" source="train" puzzle_id="f18ec8cc" lines="71-86">
<description>
Reorder the vertical color-major blocks while preserving every block's internal pixels.

Input structure
The grid is divided into vertical blocks, each with a unique dominant color.
Small marker pixels from other block colors may appear inside the blocks.

Rule
If adjacent shared-color marker rows read exactly as the current block order
from right to left, reverse the blocks. Otherwise move the leftmost block to
the far right and shift the remaining blocks left.

Key insight
The transformation acts on whole vertical blocks, so marker pixels move with
their source block instead of being recolored or independently rearranged.
</description>
</reference_puzzle>

<reference_puzzle index="2" source="train" puzzle_id="13f06aa5" lines="36-51">
<description>
Extend each colored marker from its directional sprite to a grid edge, painting a dotted ray and the whole edge side.

Input structure:
- The background is the dominant color.
- One repeated body color forms small 8-connected directional sprites.
- Each marker is a singleton non-background, non-body color embedded in one sprite.

Rule:
- The body pixels around a marker lie mostly behind it; the opposite direction is the marker's ray direction.
- Paint every second pixel from the marker along that ray, then paint the entire boundary side reached by the ray.
- If two painted boundary sides meet at a corner, make that shared corner black.

Key insight:
- The same seven-pixel body sprite rotates to encode the direction; its local center-of-mass bias points away from the ray.
</description>
</reference_puzzle>

<reference_puzzle index="3" source="train" puzzle_id="4df5b0ae" lines="27-41">
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

<reference_puzzle index="4" source="train" puzzle_id="278e5215" lines="51-66">
<description>
Recolor the large mask crop column-by-column using the separate three-row palette strip.

Input structure:
- The background is the most common color.
- One large monochrome foreground object is the mask; its bbox is the output region and may contain background holes.
- The remaining foreground is a 3-row palette rectangle with the same width as the mask bbox: two identical color rows and one uniform row.

Rule:
- Crop to the mask object's bbox.
- For each output column, color mask pixels with the corresponding color from the repeated palette rows.
- Replace background holes inside the crop with the palette's uniform third-row color.

Key insight:
The grey object is only a stencil; its columns select colors from the legend, while its holes expose the legend's fill color.
</description>
</reference_puzzle>

<reference_puzzle index="5" source="train" puzzle_id="b7955b3c" lines="386-403">
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

<reference_puzzle index="7" source="train" puzzle_id="5d588b4d" lines="10-28">
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

<reference_puzzle index="8" source="eval" puzzle_id="b0039139" lines="29-44">
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

<reference_puzzle index="9" source="train" puzzle_id="a834deea" lines="20-39">
<description>
Fill each black 5x5 region's missing interior positions with a fixed 3x3 positional color legend.

Input structure
- The grid uses teal as the dominant background and black for one or more
  5x5 regions.
- Each black region has a black one-pixel border and a 3x3 interior that
  may already contain teal pixels.

Rule
- Overlay the legend 1 7 6 / 4 8 5 / 2 9 3 on the 3x3 interior of every
  5x5 black region.
- Wherever an interior non-center legend position is still black, recolor it
  to the corresponding legend color; leave existing teal pixels, the center,
  and the black border unchanged.

Key insight
- The colored output pixels are not derived from the teal motif shape; they
  are the fixed position labels for the missing cells of each 3x3 interior.
</description>
</reference_puzzle>

<reference_puzzle index="10" source="eval" puzzle_id="62593bfd" lines="32-46">
<description>
Sort objects to top or bottom edge by whether their shape (up to rotation/flip) is shared.

Input structure:
- bg = most common color.
- Several multicolor connected components (8-connected) scattered across the grid.

Rule:
- Compute each object's canonical shape (binary mask, invariant to rotation and flip).
- Objects whose shape appears in two or more components → translate to TOP edge (row 0).
- Objects whose shape is unique → translate to BOTTOM edge (row H−h).
- Horizontal position (left column) is preserved; only vertical translation changes.

Key insight: The sorting criterion is shape frequency (rotation-invariant), not area — duplicated shapes go up, unique shapes go down.
</description>
</reference_puzzle>

<reference_puzzle index="11" source="eval" puzzle_id="21897d95" lines="283-301">
<description>
Repaint regions from T-shaped control markers, erase the markers, and optionally rotate based on a color-9 L-marker.

Input structure:
- The grid is a mosaic of monochrome regions.
- Color 1 forms T-shaped control markers on region boundaries.
- Some controls use three singleton color-1 pixels around a colored center; that center is a literal replacement color.
- An optional 5-cell L-shaped color-9 marker triggers 90-degree rotation of the output.

Rule:
- For a color-1 T marker, the single stem points at the destination region to repaint.
- The two diagonal cells behind the stem identify the source region; the destination gets that source region's original color.
- For a colored-center T marker, the same stem identifies the destination, but the center color is the replacement.
- All marker cells are filled with the final color of the source-side region.
- A 5-cell L-shaped color-9 marker controls rotation: ┐ (corner top-right) → CCW 90°, ┌ (corner top-left) → CW 90°.

Key insight:
- The stem chooses what changes, while the two diagonals behind the stem choose where the marker itself belongs.
</description>
</reference_puzzle>

<reference_puzzle index="12" source="train" puzzle_id="320afe60" lines="41-56">
<description>
Slide each foreground object horizontally to the edge indicated by whether its shape has an open notch.

Input structure:
- Yellow is the background and there is one blue foreground color.
- The foreground is split into separated 4-connected objects.
- Each object keeps its row span and shape in the output.

Rule:
- If an object's bbox border contains a background gap flanked by object pixels along that same border, slide it to the right edge and recolor it green.
- Otherwise slide the object to the left edge and recolor it red.
- Start from a blank background canvas; no moved objects overlap.

Key insight:
- The target side is encoded by the object's shape, not by its original position: open notches go right, while closed or solid shapes go left.
</description>
</reference_puzzle>

<reference_puzzle index="13" source="train" puzzle_id="981add89" lines="10-29">
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

<reference_puzzle index="14" source="train" puzzle_id="494ef9d7" lines="16-30">
<description>
Slide each blue/teal or yellow/orange row pair together while preserving its left-to-right order.

Input structure
- The grid has a black background.
- Each nonempty row contains exactly two distinct foreground pixels.
- Blue/teal and yellow/orange row pairs are the moving pairs; other color pairs are distractors.

Rule
- For each moving pair, keep the left pixel fixed and move the right pixel left until it touches the left pixel.
- Clear the moved pixel's original location, and leave non-moving rows unchanged.

Key insight
- Only same-hue-family pixel pairs attract horizontally; separated unrelated colors do not compact.
</description>
</reference_puzzle>

<reference_puzzle index="15" source="train" puzzle_id="c803e39c" lines="31-40">
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

<reference_puzzle index="16" source="train" puzzle_id="d968ffd4" lines="15-28">
<description>
Extend each of two aligned rectangular marker objects toward the other, filling half of the gap across the full perpendicular span.

Input structure
- A dominant background contains exactly two solid, same-sized monochromatic rectangles.
- The rectangles share either the same row span or the same column span, with a background gap between their facing edges.

Rule
- If the rectangles are side-by-side, color the left half of the inter-object gap across every row with the left object's color, and the right half with the right object's color.
- If one rectangle is above the other, color the upper half of the gap across every column with the upper object's color, and the lower half with the lower object's color.
- When the gap length is odd, leave the center row/column as background.

Key insight
- The marker objects are endpoints of an axial corridor; their own size identifies the facing edge, while the filled corridor spans the whole grid perpendicular to that axis.
</description>
</reference_puzzle>

<reference_puzzle index="17" source="train" puzzle_id="1be83260" lines="41-58">
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

<reference_puzzle index="18" source="train" puzzle_id="22425bda" lines="58-76">
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

<reference_puzzle index="19" source="eval" puzzle_id="dbff022c" lines="275-288">
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

<reference_puzzle index="20" source="train" puzzle_id="880c1354" lines="14-28">
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
