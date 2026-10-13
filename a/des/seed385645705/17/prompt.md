你是一个 ARC-AGI 谜题设计专家。下面的 20 道参考题是难度和机制质量的标杆。仔细阅读每道题的**解题逻辑**（解题者从样例归纳规则的认知过程），而非表面场景。你的新题应在机制深度和隐蔽性上与参考题持平或者实现超越。

每个参考题描述包含：
- `description`：一句话概述题目的变换目标。
- `Input structure`：输入网格中有哪些对象、背景、标记、边界或组件。
- `Rule`：从输入到输出的确定性变换规则。
- `Key insight`：解题所需的核心洞见。

参考题目如下：
<reference_puzzle index="1" source="train" puzzle_id="d749d46f" lines="32-51">
<description>
Pack each solid foreground rectangle twice: wide across the top and tall from the bottom.

Input structure
The grid has one background color and one foreground color; the foreground forms
separate solid rectangles, ordered from left to right.

Rule
Build a 10-row background canvas whose width is the packed width of the wide
versions of the rectangles. For each source rectangle, paint a top-aligned
rectangle of shape min(h,w) by max(h,w), with one background column between
consecutive rectangles. Then, starting again from the left edge, paint a
bottom-aligned rectangle of shape max(h,w) by min(h,w) for each source rectangle,
again separated by one background column.

Key insight
The input rectangles are not copied in their original orientations; each object
contributes one horizontal-facing and one vertical-facing normalized rectangle,
preserving left-to-right object order.
</description>
</reference_puzzle>

<reference_puzzle index="2" source="train" puzzle_id="f0f8a26d" lines="30-44">
<description>
Rotate every foreground bar 90 degrees around its own center on the same canvas.

Input structure
The background is the dominant color, and all foreground pixels share one color.
Each 4-connected foreground object is a solid one-cell-thick horizontal or vertical
bar with odd length, with single-pixel bars allowed.

Rule
Clear the canvas to the background, then redraw every bar with its midpoint, length,
and color preserved while swapping horizontal and vertical orientation.

Key insight
The bars are independent objects: their centers and lengths identify the output
placement, so no matching, ordering, or global symmetry is needed.
</description>
</reference_puzzle>

<reference_puzzle index="3" source="train" puzzle_id="3cd86f4f" lines="6-18">
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

<reference_puzzle index="5" source="train" puzzle_id="11dc524f" lines="25-40">
<description>
Slide the non-square sprite to the square anchor and replace the anchor with the sprite's mirror image.

Input structure:
- The background is the most common color.
- There is one 8-connected non-background sprite that is not a solid 2x2 square.
- There is one solid 2x2 square anchor in another non-background color.

Rule:
- Move the sprite straight toward the anchor along their unique cardinal separation until it touches the anchor's near edge.
- Reflect the moved sprite across that near edge and draw the reflected copy in the anchor color.
- Clear the original locations; the output keeps only the moved sprite and reflected anchor-color copy.

Key insight:
- The 2x2 square marks the mirror line: its side facing the sprite becomes the contact edge for a two-color reflected pair.
</description>
</reference_puzzle>

<reference_puzzle index="6" source="train" puzzle_id="f28a3cbb" lines="42-50">
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

<reference_puzzle index="7" source="train" puzzle_id="1b59e163" lines="54-69">
<description>
Copy each marked sprite to every standalone marker of the same color and erase the original sources.

Input structure:
- The background is the dominant color.
- Each source is an 8-connected component containing a monochrome sprite and one marker pixel of another color.
- Other isolated marker pixels may be destinations; marker colors with no source sprite are distractors.

Rule:
- Clear the grid to background.
- For every source marker color, place the sprite at each other marker of that color using the source marker as the anchor.
- Keep each destination marker pixel in place; do not keep source sprites, source markers, or distractor markers.

Key insight:
- A colored marker is not part of the sprite geometry; it is the anchor that says where same-colored copies of that sprite should be stamped.
</description>
</reference_puzzle>

<reference_puzzle index="8" source="train" puzzle_id="e84fef15" lines="39-59">
<description>
Reconstruct the repeated tile from its separated copies and mark the missing pixels in the damaged copy blue.

Input structure:
- The input is a lattice of equal-size blocks separated by one-cell solid rows
  and columns of the same separator color.
- All blocks are copies of one motif except exactly one block, where some
  motif pixels have been overwritten by the motif background color.
- Blue is absent from the input and is reserved as the output marker color.

Rule:
- Recover the motif by taking the unique majority color at each local block
  coordinate across all block copies.
- Find the single block that differs from this consensus motif.
- Return the consensus motif, recoloring the differing local coordinates blue.

Key insight:
- The repeated copies provide redundancy: the damaged positions are exactly
  the coordinates where the lone nonconforming copy has background instead of
  the consensus motif color.
</description>
</reference_puzzle>

<reference_puzzle index="9" source="train" puzzle_id="3d588dc9" lines="95-109">
<description>
Replace the active black cutout's tail with background and mark the cutout side facing a colored source region.

Input structure:
- The dominant color is background, with black rectangle-plus-tail cutouts, one gray source region, and colored distractors.
- Non-gray colored regions and single pixels are distractors/blockers, even when they are multi-pixel staircase shapes.

Rule:
- Find each black cutout whose largest solid rectangle leaves exactly a two-pixel tail.
- The active gray source region is visible only by horizontal rays through background from a cutout's left or right side.
- Select the unique cutout side with the most same-source visible rows, erase that cutout's tail to background, and recolor the full selected vertical side pink (6).

Key insight:
- The colored regions do not move; their horizontal line-of-sight identifies which rectangular cutout is being opened and which side receives the pink mark.
</description>
</reference_puzzle>

<reference_puzzle index="10" source="train" puzzle_id="1b8318e3" lines="86-105">
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

<reference_puzzle index="11" source="train" puzzle_id="9344f635" lines="15-30">
<description>
Extend each colored domino marker into a full row or column bar, with horizontal bars covering vertical bars at crossings.

Input structure:
- The background is the dominant color.
- Every foreground object is a 4-connected same-color domino of exactly two pixels.
- A horizontal domino marks a row to paint; a vertical domino marks a column to paint.

Rule:
- Paint each marked column entirely with the color of its vertical domino marker.
- Paint each marked row entirely with the color of its horizontal domino marker.
- Rows are painted after columns, so row bars overwrite column bars at intersections.

Key insight:
- The two-pixel orientation is the instruction: horizontal markers emit row bars, vertical markers emit column bars, and the only layering cue is row-over-column.
</description>
</reference_puzzle>

<reference_puzzle index="12" source="train" puzzle_id="22806e14" lines="23-37">
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

<reference_puzzle index="13" source="eval" puzzle_id="7666fa5d" lines="12-26">
<description>
Fill the parallelogram-shaped region between parallel anti-diagonal line segments with red.

Input structure:
- bg = most common color.
- One non-bg color forms short diagonal segments, each lying on a single anti-diagonal (r+c = const).
- Segments come in parallel pairs bounding a region to be filled.

Rule:
- For each main-diagonal axis (r−c = const), collect anti-diagonal segments whose span covers that axis.
- Sort those segments by their anti-diagonal index (r+c).
- Fill all bg cells lying strictly between consecutive segment pairs along that axis with red (color 2).

Key insight: The two diagonal directions decouple — segments sit on anti-diagonals, and filling sweeps along main diagonals between them, forming parallelogram regions.
</description>
</reference_puzzle>

<reference_puzzle index="14" source="train" puzzle_id="f0100645" lines="51-68">
<description>
Slide each inner colored object horizontally into the same-colored vertical wall until it touches an obstacle.

Input structure
- The background is the dominant color.
- The only foreground colors are two non-background colors.
- One foreground color fills the left column and the other fills the right column.
- Extra 8-connected objects of each wall color appear away from that color's wall.

Rule
- Clear the extra objects from their original positions.
- For each wall color, process its 8-connected extra objects from nearest to farthest from the matching wall.
- Slide each object horizontally toward its matching wall as far as possible without overlapping the wall or an already settled object of the same color.
- Keep the two vertical wall columns fixed.

Key insight
- The side columns act like color-specific gravity walls: same-colored objects compact sideways against their own wall while preserving each object's shape.
</description>
</reference_puzzle>

<reference_puzzle index="15" source="train" puzzle_id="78e78cff" lines="71-91">
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

<reference_puzzle index="16" source="train" puzzle_id="5034a0b5" lines="16-32">
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

<reference_puzzle index="17" source="eval" puzzle_id="a32d8b75" lines="262-275">
<description>
Assemble and place enlarged sprites on a canvas using legend-panel instructions.

Input structure:
- Color-6 separators divide the grid into a left legend column and a main canvas.
- Each legend panel has (top to bottom): a unit sprite, a frame template, a 5x5 bordered square with a position-marker pixel, and a 3x3 rotation clock-hand.

Rule:
- Tile the unit sprite into each active cell of the frame to build an enlarged composite.
- Rotate the composite per the clock-hand direction (up=0, right=90, down=180, left=270 degrees).
- Place it on the canvas anchored at the position encoded by the marker pixel's row/col in the bordered square's 3x3 inner grid.

Key insight: The marker pixel's position in a 3x3 grid jointly encodes vertical and horizontal anchor alignment (start/middle/end).
</description>
</reference_puzzle>

<reference_puzzle index="18" source="train" puzzle_id="aa62e3f4" lines="18-35">
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

<reference_puzzle index="19" source="train" puzzle_id="e3fe1151" lines="36-55">
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

<reference_puzzle index="20" source="eval" puzzle_id="2c181942" lines="7-21">
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

