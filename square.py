def pixelFromPercentage(num_pixels,percentage):
    return int((num_pixels/100)*percentage)

def get_corner(img,x_percent,y_percent):
    y_max,x_max,_ = img.shape
    x = pixelFromPercentage(x_max,x_percent)
    y = pixelFromPercentage(y_max,y_percent)
    return x,y

def get_corners(img,top_left,bottom_right):
    (x1,y1)=get_corner(img,top_left[0],top_left[1])
    (x2,y2)=get_corner(img,bottom_right[0],bottom_right[1])
    top_left = (x1,y1)
    top_right = (x2,y2)
    return top_left,top_right

def get_slice(img,top_left,bottom_right):
        top_x = top_left[0]
        top_y = top_left[1]
        bottom_x = bottom_right[0]
        bottom_y = bottom_right[1]
        return img[top_y:bottom_y,top_x:bottom_x]