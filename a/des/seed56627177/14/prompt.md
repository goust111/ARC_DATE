你是一个 ARC-AGI 谜题设计专家。下面的 20 道参考题是难度和机制质量的标杆。仔细阅读每道题的**解题逻辑**（解题者从样例归纳规则的认知过程），而非表面场景。你的新题应在机制深度和隐蔽性上与参考题持平或者实现超越。

每个参考题描述包含：
- `description`：一句话概述题目的变换目标。
- `Input structure`：输入网格中有哪些对象、背景、标记、边界或组件。
- `Rule`：从输入到输出的确定性变换规则。
- `Key insight`：解题所需的核心洞见。

参考题目如下：
<reference_puzzle index="1" source="train" puzzle_id="8618d23e" lines="9-25">
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

<reference_puzzle index="2" source="train" puzzle_id="1b8318e3" lines="86-105">
<description>
Move each isolated colored pixel into the one-cell frame around a grey 2x2 block.

Input structure:
- Background is the dominant color.
- Grey foreground objects are solid 2x2 anchor blocks.
- Every other non-background, non-grey cell is a colored singleton source.

Rule:
- Keep the grey anchors fixed and clear all colored singleton source pixels.
- For each singleton and each anchor, the normal destination is the cell reached by
  clamping the pixel into the anchor's one-cell outer frame.
- If a diagonally outside singleton has an exact 8-direction sightline into an anchor,
  also allow the cell just before the first grey pixel on that ray.
- Choose distinct destinations that minimize total king-move distance; at equal
  distance prefer exact sightlines, then the shorter Manhattan movement.

Key insight: The grey 2x2 blocks define magnetic frames; conflicts are resolved
structurally by nearest-frame assignment rather than by color identity.
</description>
</reference_puzzle>

<reference_puzzle index="3" source="train" puzzle_id="ecb67b6d" lines="138-155">
<description>
Recolor the grey diagonal-chain regions in an orange field to teal.

Input structure
The input contains only grey pixels and orange pixels.  Grey pixels form
several 8-connected regions embedded in the orange field.

Rule
Decompose each grey 8-connected region into 4-connected lobes.  Recolor the
whole region teal only when it contains a straight diagonal run of at least
three pixels and those lobes meet only by diagonal corner contacts in a
non-branching diagonal chain or cycle; a single 4-connected region is
recolored only when it is a square diagonal staircase.

Key insight
The teal regions are not selected by size or position; they are the grey
regions whose internal connectivity depends on diagonal corner joins, while
same-edge cracks, rectangles, bars, and branching/noisy regions remain grey.
</description>
</reference_puzzle>

<reference_puzzle index="4" source="train" puzzle_id="fd02da9e" lines="26-41">
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

<reference_puzzle index="5" source="train" puzzle_id="b25e450b" lines="28-43">
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

<reference_puzzle index="6" source="eval" puzzle_id="38007db0" lines="9-22">
<description>
Extract the unique cell from each lattice row into a single-column lattice.

Input structure:
- Grid divided into equal square cells by uniform gridlines of color L.
- Cell interiors filled with bg; each cell has an inner sprite pattern.
- Each row of cells has exactly one cell with a unique pattern; the rest are copies.

Rule:
- For each lattice row, find the cell whose inner pattern appears exactly once (the "odd one out").
- Construct a single-column lattice preserving the original row heights, containing only the unique cells.

Key insight: The task is per-row "odd one out" — every cell in a row is identical except one oddball, which must be extracted.
</description>
</reference_puzzle>

<reference_puzzle index="7" source="train" puzzle_id="825aa9e9" lines="111-125">
<description>
Drop the moving-color objects downward until they rest above a one-row cushion before the obstacle or bottom edge.

Input structure:
- The grid has exactly three colors: empty background, a stationary obstacle/floor color, and a moving object color.
- The obstacle color is the unique non-background role that touches the bottom edge.
- Moving pixels start outside the obstacle cells, outside the row directly above each obstacle pixel, and outside the bottom row.

Rule:
- Keep the background and obstacle pixels fixed.
- Treat obstacle pixels, cells directly above obstacle pixels, and the bottom row as blocked.
- Repeatedly move every 4-connected moving-color object one row down when its translated shape does not hit blocked cells or another moving object.

Key insight: The visible floor has an invisible one-row cushion above it, so the objects fall under gravity but stop with one empty row before the floor or the grid bottom.
</description>
</reference_puzzle>

<reference_puzzle index="8" source="train" puzzle_id="d968ffd4" lines="15-28">
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

<reference_puzzle index="9" source="train" puzzle_id="412b6263" lines="39-54">
<description>
Infer the background, rotate the whole input 90 degrees counterclockwise, and stack two framed copies vertically.

Input structure:
- Blue is absent from the input.
- One rare marker color appears in exactly two pixels.
- The background is the unique non-marker color whose removal leaves exactly two 4-connected foreground objects, each with one marker pixel and one body color.
- The whole grid, including background and foreground objects, is the sprite to transform.

Rule:
- Rotate the entire input grid 90 degrees counterclockwise.
- Create a blue one-pixel frame around two vertical copies of the rotated grid.
- Use the input background color for the two endpoints of every horizontal frame/separator row.

Key insight: The output is not a crop of the foreground objects; it is two copies of the complete rotated input, separated and enclosed by a newly added blue border.
</description>
</reference_puzzle>

<reference_puzzle index="10" source="train" puzzle_id="320afe60" lines="41-56">
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

<reference_puzzle index="11" source="train" puzzle_id="f341894c" lines="63-78">
<description>
Reorient each two-pixel blue/pink segment so it reads pink then blue when viewed from its border marker.

Input structure
- The background is dominant, with one border-only marker color and blue/pink foreground segments.
- Every marker lies on the outer border and points inward along its row or column.
- Along exactly one inward ray from each marker is a single adjacent two-pixel blue/pink segment.

Rule
- For each marker, look inward perpendicular to the border side containing the marker.
- Rewrite the two blue/pink pixels on that ray so the nearer one is pink and the farther one is blue.
- Keep marker pixels, background pixels, and all other pixels unchanged.

Key insight
- The marker gives a local viewing direction; the segment's required order is relative to that marker, not to global row or column order.
</description>
</reference_puzzle>

<reference_puzzle index="12" source="train" puzzle_id="ad173014" lines="97-111">
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

<reference_puzzle index="13" source="train" puzzle_id="ac0c2ac3" lines="7-19">
<description>
Fill each centered square ring with the color of the marker at that ring distance.

Input structure
The input is an odd square grid with a dominant background color and one
non-background marker for every Chebyshev distance from the grid center.
Rule
The output replaces the background with concentric square rings centered on
the middle cell; every cell at distance d from the center takes the color of
the unique marker whose distance from the center is d.
Key insight
The scattered markers are not objects to move; their Chebyshev distances
form a palette for the centered square rings.
</description>
</reference_puzzle>

<reference_puzzle index="14" source="eval" puzzle_id="58490d8a" lines="9-21">
<description>
Count colored objects and record tallies inside a legend box.

Input structure:
- A black rectangular legend box with key-color pixels in its leftmost non-black column.
- Outside the box, distinct objects of each key color are scattered across the grid.

Rule:
- For each key color, count separate same-colored objects outside the legend box.
- Output the legend box repainted: each key row gets one dot per object, spaced every other cell rightward from the key pixel.

Key insight: The legend is both the color index and the output canvas — object counts become dot sequences on the matching row.
</description>
</reference_puzzle>

<reference_puzzle index="15" source="train" puzzle_id="f8f52ecc" lines="107-122">
<description>
Connect same-colored foreground components with one-cell-wide orthogonal paths around teal wall segments.

Input structure
- The dominant color is the background.
- Teal cells are fixed wall segments and are never source connectors.
- Other foreground colors appear as one or more 4-connected components; colors with a single component stay unchanged.

Rule
- For each non-background, non-teal color, read its 4-connected components in row-major order.
- Connect each consecutive pair by the shortest clear L-shaped path between the two components.
- Clear paths may pass only through background or the same color, and they preserve the previous travel axis when both bends are possible.

Key insight
- The output is not a global recolor: each color traces a routed chain through its own components while teal cells and other colors act as blockers.
</description>
</reference_puzzle>

<reference_puzzle index="16" source="train" puzzle_id="df978a02" lines="57-72">
<description>
Let the largest center-pointing object grow outward while every other object loses its inward tip.

Input structure
- The grid has a single dominant background color and four 8-connected monochromatic foreground objects.
- The objects surround a central background gap and point toward it from cardinal directions.
- Exactly one object has the largest area.

Rule
- Find the central background gap by minimizing aggregate distance to the four objects.
- Recolor the nearest-to-center pixel of each non-largest object to background.
- Add a three-pixel cap in the largest object's color on the side away from the center, centered on the shared row or column.

Key insight
- The edit is a transfer of prominence around a common center: losers retract at their tips, and the unique largest object extends outward from the center.
</description>
</reference_puzzle>

<reference_puzzle index="17" source="train" puzzle_id="6350f1f4" lines="62-81">
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

<reference_puzzle index="18" source="train" puzzle_id="1be83260" lines="41-58">
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

<reference_puzzle index="19" source="train" puzzle_id="abbfd121" lines="56-74">
<description>
Recover the repeating background pattern hidden under the largest solid rectangle.

Input structure:
- The grid is covered by a small row/column periodic pattern.
- Several non-pattern colors form solid rectangular foreground objects that
  overwrite the pattern.
- The requested output region is the largest of those solid rectangles.

Rule:
- Find every 4-connected monochromatic object whose bounding box is a filled
  rectangle with both dimensions greater than one.
- Treat those rectangles as masks, infer the smallest repeating tile that is
  consistent with all remaining visible pixels, and repaint the largest
  rectangle's coordinates with that tile.

Key insight: The rectangle colors are occluders; the visible pixels outside
them are enough to reconstruct the original periodic background underneath.
</description>
</reference_puzzle>

<reference_puzzle index="20" source="train" puzzle_id="18447a8d" lines="33-48">
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

