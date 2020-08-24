from gl import Render, color, V2, V3
from obj import Obj, Texture

from shaders import *

r = Render(768,432)

r.active_texture = Texture('./models/model.bmp')
r.active_shader = phong

posModel = V3( 0, 0, -5)

r.lookAt(posModel, V3(2,2,0))

r.loadModel('./models/model.obj', posModel, V3(1,1,1), V3(0,0,0))

r.glFinish('output.bmp')
