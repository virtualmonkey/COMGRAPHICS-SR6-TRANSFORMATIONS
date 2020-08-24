import numpy as np
from collections import namedtuple

from obj import  Obj
from utils.gl_color import color, decimalToRgb
from utils.gl_encode import char, word, dword
from utils.gl_math import cross, dot, substract, norm, V2, V3

BLACK = color(0,0,0)
WHITE = color(1,1,1)

def baryCoords(A, B, C, P):
    # u es para la A, v es para B, w para C
    try:
        u = ( ((B.y - C.y)*(P.x - C.x) + (C.x - B.x)*(P.y - C.y) ) /
              ((B.y - C.y)*(A.x - C.x) + (C.x - B.x)*(A.y - C.y)) )

        v = ( ((C.y - A.y)*(P.x - C.x) + (A.x - C.x)*(P.y - C.y) ) /
              ((B.y - C.y)*(A.x - C.x) + (C.x - B.x)*(A.y - C.y)) )

        w = 1 - u - v
    except:
        return -1, -1, -1

    return u, v, w


class Render(object):
    def __init__(self, width, height):
        self.curr_color = WHITE
        self.clear_color = BLACK
        self.glCreateWindow(width, height)

        self.light = V3(0,0,1)
        self.active_texture = None

    def glCreateWindow(self, width, height):
        self.width = width
        self.height = height
        self.glClear()
        self.glViewport(0,0, width, height)

    def glViewport(self, x, y, width, height):
        self.viewport_initial_x = x
        self.viewport_initial_y = y
        self.viewport_width = width
        self.viewport_height = height
        self.viewport_final_x = x + width
        self.viewport_final_y = x + height

    def glClear(self):
        self.pixels = [ [ self.clear_color for x in range(self.width)] for y in range(self.height) ]
    
        # Zbuffer
        self.zbuffer = [ [ -float('inf') for x in range(self.width)] for y in range(self.height) ]
        
    def glVertextInViewport(self, x,y):
        return (x >= self.viewport_initial_x and
            x <= self.viewport_final_x) and (
            y >= self.viewport_initial_y and
            y <= self.viewport_final_y)

    def glClearColor(self, r,g,b):
        rgb_array = decimalToRgb([r,g,b])
        self.clear_color = color(rgb_array[0], rgb_array[1], rgb_array[2])

    def glVertex(self, x, y, color = None):
        pixelX = ( x + 1) * (self.vpWidth  / 2 ) + self.vpX
        pixelY = ( y + 1) * (self.vpHeight / 2 ) + self.vpY
        try:
            self.pixels[round(pixelY)][round(pixelX)] = color or self.curr_color
        except:
            pass
    
    def glPoint(self, x, y, color = None):
        if x >= self.width or x < 0 or y >= self.height or y < 0:
            return
        try:
            self.pixels[y][x] = color or self.curr_color
        except:
            pass
    
    def glColor(self, r,g,b):
        rgb_array = decimalToRgb([r,g,b])
        self.curr_color = color(rgb_array[0], rgb_array[1], rgb_array[2])

    def glFixCoordinate(self, value, main_axis):
        fixed_coordinate = 0
        if main_axis:
            fixed_coordinate = (value+1) * (self.viewport_width/2) + self.viewport_initial_x
        else:
            fixed_coordinate = (value+1) * (self.viewport_height/2) + self.viewport_initial_y
        return round(fixed_coordinate)
    
        
    # Generate .bmp fil
    def glFinish(self, filename):
        archivo = open(filename, 'wb')

        # File header 14 bytes
        #archivo.write(char('B'))
        #archivo.write(char('M'))

        archivo.write(bytes('B'.encode('ascii')))
        archivo.write(bytes('M'.encode('ascii')))

        archivo.write(dword(14 + 40 + self.width * self.height * 3))
        archivo.write(dword(0))
        archivo.write(dword(14 + 40))

        # Image Header 40 bytes
        archivo.write(dword(40))
        archivo.write(dword(self.width))
        archivo.write(dword(self.height))
        archivo.write(word(1))
        archivo.write(word(24))
        archivo.write(dword(0))
        archivo.write(dword(self.width * self.height * 3))
        archivo.write(dword(0))
        archivo.write(dword(0))
        archivo.write(dword(0))
        archivo.write(dword(0))
        
        # Pixeles, 3 bytes cada uno

        for x in range(self.height):
            for y in range(self.width):
                archivo.write(self.pixels[x][y])

        archivo.close()

    def glZBuffer(self, filename):
        archivo = open(filename, 'wb')

        # File header 14 bytes
        archivo.write(bytes('B'.encode('ascii')))
        archivo.write(bytes('M'.encode('ascii')))
        archivo.write(dword(14 + 40 + self.width * self.height * 3))
        archivo.write(dword(0))
        archivo.write(dword(14 + 40))

        # Image Header 40 bytes
        archivo.write(dword(40))
        archivo.write(dword(self.width))
        archivo.write(dword(self.height))
        archivo.write(word(1))
        archivo.write(word(24))
        archivo.write(dword(0))
        archivo.write(dword(self.width * self.height * 3))
        archivo.write(dword(0))
        archivo.write(dword(0))
        archivo.write(dword(0))
        archivo.write(dword(0))

        # Minimo y el maximo
        minZ = float('inf')
        maxZ = -float('inf')
        for x in range(self.height):
            for y in range(self.width):
                if self.zbuffer[x][y] != -float('inf'):
                    if self.zbuffer[x][y] < minZ:
                        minZ = self.zbuffer[x][y]

                    if self.zbuffer[x][y] > maxZ:
                        maxZ = self.zbuffer[x][y]

        for x in range(self.height):
            for y in range(self.width):
                depth = self.zbuffer[x][y]
                if depth == -float('inf'):
                    depth = minZ
                depth = (depth - minZ) / (maxZ - minZ)
                archivo.write(color(depth,depth,depth))

        archivo.close()
    def glLine(self, v0, v1, color = None) :
        x0 = self.glFixCoordinate(v0.x, True)
        x1 = self.glFixCoordinate(v1.x, True)
        y0 = self.glFixCoordinate(v0.y, False)
        y1 = self.glFixCoordinate(v1.y, False)

        steep = abs(y1 - y0) > abs(x1 - x0)

        if steep:
            x0, y0 = y0, x0
            x1, y1 = y1, x1
        if x0 > x1:
            x0, x1 = x1, x0
            y0, y1 = y1, y0

        dx, dy = abs(x1 - x0), abs(y1 - y0)      
        
        offset = 0
        limit =  0.5
        y = y0

        for x in range(x0, x1+1):
            self.glPoint(y, x, color) if steep else self.glPoint(x, y, color)
            
            offset += 2*dy

            if offset >= limit:
                y += 1 if y0 < y1 else -1
                limit += 2*dx

    def glLine_coord(self, v0, v1, color = None):
        x0 = v0.x
        x1 = v1.x
        y0 = v0.y
        y1 = v1.y

        steep = abs(y1 - y0) > abs(x1 - x0)

        if steep:
            x0, y0 = y0, x0
            x1, y1 = y1, x1
        if x0 > x1:
            x0, x1 = x1, x0
            y0, y1 = y1, y0

        dx, dy = abs(x1 - x0), abs(y1 - y0)      
        
        offset = 0
        limit =  0.5
        y = y0
    
        try:
            m = dy/dx
        except ZeroDivisionError:
            pass
            
        for x in range(x0, x1+1):
            self.glPoint(y, x, color) if steep else self.glPoint(x, y, color)
            
            offset += 2*dy

            if offset >= limit:
                y += 1 if y0 < y1 else -1
                limit += 2*dx        
    
    def transform(self, vertex, translate=V3(0,0,0), scale=V3(1,1,1)):
        return V3(round(vertex[0] * scale.x + translate.x),
                  round(vertex[1] * scale.y + translate.y),
                  round(vertex[2] * scale.z + translate.z))
    
    # Check if a given point (x,y) is inside the polygon
    def glIsPointInPolygon(self, x, y, polygon):
        # Args:
        #   x: the x coordinate of point.
        #   y: the y coordinate of point.
        #   polygon: a list of tuples [(x, y), (x, y), ...] representing the vertices of the polygon

        # Returns:
        #   True if the point is in the path.
        verticesCount = len(polygon)
        j = verticesCount - 1
        c = False
        for i in range(verticesCount):
            if ((polygon[i][1] > y) != (polygon[j][1] > y)) and \
                    (x < polygon[i][0] + (polygon[j][0] - polygon[i][0]) * (y - polygon[i][1]) /
                                    (polygon[j][1] - polygon[i][1])):
                c = not c
            j = i
        return c
    
    # Fill the polygon
    def glFillPolygon(self, polygon):
        # Args:
        #   polygon: a list of tuples [(x, y), (x, y), ...] representing the vertices of the polygon

        # Returns:
        #   nothing
        
        minX, maxX, minY, maxY = 0,0,0,0
        
        # Calculate the min and max points in x-axis and y-axis for the polygon
        for i in range(len(polygon)):
            if(polygon[i][0] < minX):
                minX = polygon[i][0]
            elif(polygon[i][0] > maxX):
                maxX = polygon[i][0]
            if(polygon[i][1] < minY):
                minY = polygon[i][1]
            elif(polygon[i][1] > maxY):
                maxY = polygon[i][1]

        # Iterate over those numbers and check if every point is in the polygon
        # If it is, fill it
        for y in range(minY, maxY):
            for x in range(minX, maxX):
                if (self.glIsPointInPolygon(x,y, polygon)):
                    self.glPoint(x, y)
    
    # Draw the polygon joining the dots with glLinge_coord
    def glDrawPolygon(self, vertices):
        count = len(vertices)

        for limit in range(count):
            v0 = vertices[limit]
            v1 = vertices[(limit + 1) % count]
            self.glLine_coord(v0[0], v0[1], v1[0], v1[1])


    def loadModel(self, filename, translate = V3(0,0,0), scale = V3(1,1,1), isWireframe = False):
        model = Obj(filename)

        light = V3(0,0,1)

        for face in model.faces:

            vertCount = len(face)

            if isWireframe:
                for vert in range(vertCount):
                    v0 = model.vertices[ face[vert][0] - 1 ]
                    v1 = model.vertices[ face[(vert + 1) % vertCount][0] - 1]
                    v0 = V2(round(v0[0] * scale.x  + translate.x),round(v0[1] * scale.y  + translate.y))
                    v1 = V2(round(v1[0] * scale.x  + translate.x),round(v1[1] * scale.y  + translate.y))
                    self.glLine_coord(v0, v1)

            else:
                v0 = model.vertices[ face[0][0] - 1 ]
                v1 = model.vertices[ face[1][0] - 1 ]
                v2 = model.vertices[ face[2][0] - 1 ]
                if vertCount > 3:
                    v3 = model.vertices[ face[3][0] - 1 ]

                v0 = self.transform(v0,translate, scale)
                v1 = self.transform(v1,translate, scale)
                v2 = self.transform(v2,translate, scale)
                if vertCount > 3:
                    v3 = self.transform(v3,translate, scale)

                if self.active_texture:
                    vt0 = model.texcoords[face[0][1] - 1]
                    vt1 = model.texcoords[face[1][1] - 1]
                    vt2 = model.texcoords[face[2][1] - 1]
                    vt0 = V2(vt0[0], vt0[1])
                    vt1 = V2(vt1[0], vt1[1])
                    vt2 = V2(vt2[0], vt2[1])
                    if vertCount > 3:
                        vt3 = model.texcoords[face[3][1] - 1]
                        vt3 = V2(vt3[0], vt3[1])
                else:
                    vt0 = V2(0,0) 
                    vt1 = V2(0,0) 
                    vt2 = V2(0,0) 
                    vt3 = V2(0,0)

                # vn0 = model.normals[face[0][2] - 1]
                # vn1 = model.normals[face[1][2] - 1]
                # vn2 = model.normals[face[2][2] - 1]
                # if vertCount > 3:
                #     vn3 = model.normals[face[3][2] - 1]

                normal = cross(substract(v1,v0), substract(v2,v0))
                intensity = dot(norm(normal), norm(light))

                if intensity >=0:
                    self.triangle_bc(v0,v1,v2, texture=self.active_texture, texcoords=(vt0,vt1, vt2), intensity=intensity)

                    # Manage square rendering
                    if vertCount > 3:
                        v3 = model.vertices[ face[3][0] - 1 ]
                        v3 = self.transform(v3,translate, scale)
                        if intensity >=0:
                            self.triangle_bc(v0,v2,v3, color(intensity, intensity, intensity))

    #Barycentric Coordinates
    def triangle_bc(self, A, B, C, texture, _color= WHITE,texcoords = (), intensity = 1):
        #bounding box
        minX = min(A.x, B.x, C.x)
        minY = min(A.y, B.y, C.y)
        maxX = max(A.x, B.x, C.x)
        maxY = max(A.y, B.y, C.y)

        for x in range(minX, maxX + 1):
            for y in range(minY, maxY + 1):
                if x >= self.width or x < 0 or y >= self.height or y < 0:
                    continue

                u, v, w = baryCoords(A, B, C, V2(x, y))

                if u >= 0 and v >= 0 and w >= 0:

                    z = A.z * u + B.z * v + C.z * w
                    if z > self.zbuffer[y][x]:

                        b, g, r = _color
                        b/= 255
                        g/= 255
                        r/= 255

                        b*= intensity
                        g*= intensity
                        r*= intensity

                        if texture:
                            ta, tb, tc = texcoords

                            tx = ta.x * u + tb.x * v + tc.x*w
                            ty = ta.y * u + tb.y * v + tc.y*w

                            texColor = texture.getColor(tx,ty)
                            b*=texColor[0] /255
                            g*=texColor[1] /255
                            r*=texColor[2] /255

                        # r, g, b = self.active_shader(
                        #     self,
                        #     verts=(A,B,C),
                        #     baryCoords=(u,v,w),
                        #     texCoords=texcoords,
                        #     normals=normals,
                        #     color = _color or self.curr_color)

                        self.glPoint(x, y, color(r,g,b))
                        self.zbuffer[y][x] = z








