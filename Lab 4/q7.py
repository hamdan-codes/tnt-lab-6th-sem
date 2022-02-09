a, b, c = map(int, input('Enter a, b, c: ').split())

d = b*b - 4*a*c

if d < 0:
    print('Imaginary roots')
elif d == 0:
    r = -b / (2*a)
    print('Real and equal roots:', r)
else:
    r1 = (-b + (d**0.5)) / (2*a)
    r2 = (-b - (d**0.5)) / (2*a)
    print('Real and different roots:', r1, 'and', r2)
