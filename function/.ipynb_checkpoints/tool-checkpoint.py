def slope(x1, y1, x2, y2):
    m = 0
    d = (x1 - x2)
    b = (y1 - y2)
    if b != 0:
        m = round((d)/(b), 5) 
    return m