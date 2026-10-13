你是一个 ARC-AGI 谜题设计专家。下面的 20 道参考题是难度和机制质量的标杆。仔细阅读每道题的**解题逻辑**（解题者从样例归纳规则的认知过程），而非表面场景。你的新题应在机制深度和隐蔽性上与参考题持平或者实现超越。

每个参考题描述包含：
- `description`：一句话概述题目的变换目标。
- `Input structure`：输入网格中有哪些对象、背景、标记、边界或组件。
- `Rule`：从输入到输出的确定性变换规则。
- `Key insight`：解题所需的核心洞见。

参考题目如下：
<reference_puzzle index="1" source="train" puzzle_id="8a6d367c" lines="52-67">
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

<reference_puzzle index="2" source="train" puzzle_id="25e02866" lines="14-31">
<description>
Merge same-sized filled rectangular objects by overlaying their internal marker pixels.

Input structure:
- A dominant background surrounds two or more separated 4-connected foreground objects.
- Every object is a filled rectangle of the same size.
- Each rectangle has the same majority base color plus one or more non-base marker pixels.

Rule:
- Create one output rectangle with the shared object size, filled with the shared base color.
- For each input rectangle, copy every non-base marker pixel to the same relative location
  in the output.
- Marker overlays must either land on base color or agree in color with an existing marker.

Key insight:
The separated rectangles are layers of the same template; their colored marker pixels
combine by local coordinates while the repeated base fill is just the output canvas.
</description>
</reference_puzzle>

<reference_puzzle index="3" source="train" puzzle_id="ad38a9d0" lines="23-36">
<description>
Recolor each placeholder object according to its normalized shape while leaving the background fixed.

Input structure
A same-size grid with one dominant background color and one foreground
placeholder color forming separated 4-connected objects.

Rule
Replace every placeholder object by the color assigned to its exact
translation-normalized mask; all background pixels remain unchanged.

Key insight
The input objects are a shape legend with repeated instances: identical
masks always receive identical colors, independent of their position.
</description>
</reference_puzzle>

<reference_puzzle index="4" source="train" puzzle_id="f8cc533f" lines="10-25">
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

<reference_puzzle index="5" source="train" puzzle_id="83eb0a57" lines="72-86">
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

<reference_puzzle index="6" source="eval" puzzle_id="446ef5d2" lines="8-20">
<description>
Assemble scattered rectangular pieces into a rectangle at an L-marker corner.

Input structure:
- bg = most common color. Non-bg rectangular pieces scattered on the grid.
- 3-pixel L-marker (2x2 minus one cell) tags the corner; the gap is the corner cell.

Rule:
- rect_col = most common color across all piece pixels; all outer rectangle edges must be rect_col.
- Translate pieces (no rotation) to tile the rectangle; touching edges must match, with rect_col as wildcard.

Key insight: rect_col acts as wildcard during edge matching — only non-rect_col pixels at shared boundaries must agree.
</description>
</reference_puzzle>

<reference_puzzle index="7" source="train" puzzle_id="da6e95e5" lines="70-81">
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

<reference_puzzle index="8" source="eval" puzzle_id="38007db0" lines="9-22">
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

<reference_puzzle index="9" source="train" puzzle_id="8e301a54" lines="8-23">
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

<reference_puzzle index="11" source="train" puzzle_id="f1bcbc2c" lines="40-55">
<description>
Paint the top-connected middle channel between two orange path components teal.

Input structure
- The background is black.
- One foreground color forms exactly two 4-connected one-pixel-wide path components.
- Optional non-background control pixels may sit in the channel between the two paths.

Rule
- Dilate each path component by one cell in the 8-neighborhood and intersect the two dilations; this gives the one-cell-wide middle channel.
- A control pixel in that channel is a blocker only when its 4-neighbors touch fewer than both path components; otherwise it is part of the channel and is repainted.
- After removing blockers, paint the channel component connected to the top edge teal and keep all other pixels unchanged.

Key insight
- The desired pixels are not found by region filling; they are the geometric overlap of the two paths' one-cell halos, with weakly attached control pixels splitting the channel.
</description>
</reference_puzzle>

<reference_puzzle index="12" source="eval" puzzle_id="2c181942" lines="7-21">
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

<reference_puzzle index="13" source="eval" puzzle_id="898e7135" lines="7-20">
<description>
Inflate a frame's hole-layout and fill each hole with its matching piece.

Input structure:
- bg = most common color. Largest component is a monochrome frame enclosing bg holes.
- Outside pieces composed of k×k blocks; k = side of smallest solid square piece.

Rule:
- Extract connected hole regions inside the frame as slots.
- Match each piece to a slot by comparing block-level shapes under rotation/flip.
- Output the layout scaled by k, with pieces placed in slots; frame color as background.

Key insight: Pieces are k×k-block-scaled versions of holes; downsampling reveals shape matches under all 8 orientations.
</description>
</reference_puzzle>

<reference_puzzle index="14" source="train" puzzle_id="f28a3cbb" lines="42-50">
<description>
Compress each colored ray pixel inward until it touches that color's 3x3 anchor block.
Input structure
- A uniform background grid with non-background colors.
- Each foreground color contains one solid 3x3 anchor block; all other pixels of that color lie on straight 8-neighbor rays aimed at that block.
Rule
- Keep anchor blocks fixed, erase loose foreground pixels, and for each ray pack its pixels into consecutive cells adjacent to the anchor along the same ray.
Key insight
- The scattered pixels are displaced pixels on visible row, column, or diagonal sightlines back to their same-colored 3x3 block.
</description>
</reference_puzzle>

<reference_puzzle index="15" source="train" puzzle_id="17b866bd" lines="49-67">
<description>
Move each marker from a lattice corner gap into the adjacent enclosed 12-pixel cavity.

Input structure:
- The grid is a repeated teal lattice around black 12-pixel cavities.
- A marker is a non-black pixel placed at the northwest corner gap of one
  complete 5x5 lattice tile; the marker may itself be teal.

Rule:
- For every marked northwest corner gap, clear that marker to black.
- Fill the 12 black pixels in the tile immediately down-right of the marker
  with the marker color.
- Leave the rest of the lattice unchanged.

Key insight:
- The apparent isolated markers are control pixels for the adjacent cavity;
  teal markers are recognized as extra teal pixels occupying a gap that the
  periodic lattice template says should be black.
</description>
</reference_puzzle>

<reference_puzzle index="16" source="eval" puzzle_id="2d0172a1" lines="209-224">
<description>
Convert a rough nested line drawing into a clean canonical frame-and-dot diagram.

Input structure:
- The grid has one background color and one foreground color.
- Foreground 4-connected objects are rough closed frames or small filled blobs.
- Frames may contain other frames/blobs; blobs outside a frame remain outside it.

Rule:
- Detect the containment tree of foreground objects using holes of each frame object.
- Replace every blob with one foreground pixel.
- Replace every frame with a rectangular border padded two cells around its direct children.
- Place siblings by their original row/column overlap and order, with one empty cell between canonical objects.

Key insight: The exact wobbly geometry is irrelevant; only enclosure plus relative sibling positions define the output.
</description>
</reference_puzzle>

<reference_puzzle index="17" source="train" puzzle_id="9f5f939b" lines="42-57">
<description>
Mark the missing center of each plus-shaped arrangement of four two-pixel bars with yellow.

Input structure:
- The background is the dominant color.
- The single foreground color forms isolated 4-connected objects, each a horizontal or vertical two-pixel bar.
- Some bars form an implied plus sign: two horizontal bars and two vertical bars around an empty center, with a one-pixel gap between the center and each bar.

Rule:
- Keep the input unchanged except at implied plus-sign centers.
- For every background pixel with foreground bars at offsets 2-3 cells left, right, above, and below, recolor that center yellow.
- The one-pixel gaps immediately next to the center in the four cardinal directions must remain background.

Key insight:
- The yellow pixels are not extending a line; they identify the shared missing center where four separated two-pixel bars align as a plus sign.
</description>
</reference_puzzle>

<reference_puzzle index="18" source="train" puzzle_id="cc9053aa" lines="59-72">
<description>
Recolor the unique safe route between the two maroon endpoints from teal to maroon.

Input structure
- A black background contains a teal 4-connected track, exactly two maroon endpoint pixels, and one or more orange blocker pixels embedded near the track.

Rule
- Treat each orange blocker and its 8-neighborhood as forbidden.
- In the remaining teal/maroon track graph, there is exactly one simple
  4-connected path between the maroon endpoints.
- Recolor teal pixels on that path to maroon; leave endpoints, blockers, and all other cells unchanged.

Key insight
- The orange pixels block nearby teal cells diagonally as well as orthogonally, which removes shorter decoy routes and makes the intended endpoint-to-endpoint path unique.
</description>
</reference_puzzle>

<reference_puzzle index="19" source="train" puzzle_id="ac0c2ac3" lines="7-19">
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

<reference_puzzle index="20" source="train" puzzle_id="17829a00" lines="21-36">
<description>
Move each colored object vertically until it touches the matching top or bottom border bar.

Input structure:
- The background is the dominant color.
- The top row is a full-width bar in one foreground color and the bottom row is a full-width bar in another.
- All other foreground pixels are 8-connected monochrome objects whose color matches one of those bars.

Rule:
- Top-bar-colored objects move upward so their bbox top is row 1.
- Bottom-bar-colored objects move downward so their bbox bottom is the row above the bottom bar.
- A downward-moving one-column segment whose landing overlaps its source becomes the continuous vertical segment swept between them.

Key insight:
The horizontal bars are destination anchors: color determines which border an object slides toward, while columns and object shape are otherwise preserved.
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

请一次性生成5个`<new_puzzle>`，若有较多合适的创新，也可以生成更多，不设上限，但不要输出额外解释。

注意事项：
1. 任务不应依赖大量先验知识或复杂计算，而应依赖对模式、结构与空间关系的直觉推理。
2. 关于"计算"的边界：
   - **允许使用**（这些都是人类能心算的视觉量）：数像素 / 孔洞 / 对象 / 邻居（≤10 量级）、比大小、按某个可见属性排序、奇偶判断、取最大 / 最小、用一个计数值作为另一个等量几何量（如 N 个像素 → 长度 N 的射线、面积 N → 边长 N）、判断接触 / 相邻 / 包含、粗略中心位置（簇的视觉中点）、"距离"作为"几步"而不是公式等人类能够直观感受计算的内容。
   - **禁止使用**（这些是算法量而非视觉量，会迫使解题者依赖公式而非视觉直觉）：曼哈顿 / 欧氏 / Chebyshev 距离作为精确查表键、凸包面积、二进制位编码 / 位运算、模运算（取模） / 余数、拓扑距离（BFS / 最短路 / 跳数）、精确几何质心坐标、凸角 / 凹角计数等复杂且不易被看出来的计算都是禁止使用的。
3. 问题必须能从少量示例中推断出来（通常为 3 对训练样例 + 1 个测试样例）。设计时从人类解题者的视觉角度思考：规则涉及的所有关键要素（对象、关系、属性）必须在网格上肉眼可见，不应依赖人眼无法直接观察到的隐含信息（例如颜色id和颜色亮度是禁止使用的）或需要推算的隐藏属性。规则背后的逻辑本身可以有一定挑战性，但解题所需的所有线索必须能从网格中直接获取。
4. 生成的问题描述旨在促进后续用于创建题目网格的代码生成。因此，对输入 / 输出网格的描述以及规则总结必须极其详细；语言风格应易于理解；对于特定对象（如箭头），必须在描述中明确其绘制方式。
5. 描述谜题时避免使用 0 或 1 这类颜色编号；应使用颜色名称，例如 "red"。可用颜色如下：Black、Blue、Red、Green、Yellow、Grey、Magenta、Orange、Azure、Maroon。
6. 新题的核心推理机制应明显不同——即从观察样例到归纳规则的认知路径不同，而不仅仅是场景、对象或颜色不同。
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

