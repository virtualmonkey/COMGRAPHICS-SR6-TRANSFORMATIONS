from gl import Render, color, V2, V3
from obj import Obj, Texture

from shaders import *

r = Render(768,432)

r.active_texture = Texture('./models/model.bmp')
r.active_shader = phong

posModel = V3( 0, 0, -5)

# Medium shot

r.lookAt(posModel, V3(0,-1,0))

r.loadModel('./models/model.obj', posModel, V3(1,1,1), V3(0,0,0))

r.glFinish('medium.bmp')

# Low angle

r = Render(768,432)

r.active_texture = Texture('./models/model.bmp')
r.active_shader = phong

posModel = V3( 0, 0, -5)

r.lookAt(posModel, V3(0,-5,0))

r.loadModel('./models/model.obj', posModel, V3(1,1,1), V3(0,0,0))

r.glFinish('low.bmp')


# High angle

r = Render(768,432)

r.active_texture = Texture('./models/model.bmp')
r.active_shader = phong

posModel = V3( 0, 0, -5)

posModel = V3( 0, 0, -3)

r.lookAt(posModel, V3(0,4,0))

r.loadModel('./models/model.obj', posModel, V3(1,1,1), V3(0,0,0))

r.glFinish('high.bmp')

# Dutch angle

r = Render(768,432)

r.active_texture = Texture('./models/model.bmp')
r.active_shader = phong

posModel = V3( 0, 0, -5)

r.lookAt(posModel, V3(0,1,0))

r.loadModel('./models/model.obj', posModel, V3(1,1,1), V3(0,0,66))

r.glFinish('dutch.bmp')
