from gl import Render, color, V2, V3
from obj import Obj, Texture

import random

r = Render(1000,1000)

r.active_texture = Texture('./models/model.bmp')

r.loadModel('./models/model.obj', V3(500,500,0), V3(300,300,300))

r.glFinish('output.bmp')
#r.glZBuffer('zbuffer.bmp')
