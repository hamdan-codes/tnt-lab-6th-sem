gd, bd = 3, 5
ng, nb = 0, 0

for i in range(gd):
    ng += int(input(f'Girls in {i+1} grp: '))
for i in range(bd):
    nb += int(input(f'Boys in {i+1} grp: '))

print('Total girls =', ng)
print('Total boys =', nb)
print('Total students =', (ng+nb))
