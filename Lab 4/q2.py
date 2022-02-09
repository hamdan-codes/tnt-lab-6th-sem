p, c, e, m = 92, 72, 83, 65

p += 5
c += 5

def grade(m):
    ans = 'F'
    if m >= 90:
        ans = 'O'
    elif m >= 80:
        ans = 'E'
    elif m >= 70:
        ans = 'A'
    elif m >= 60:
        ans = 'B'
    elif m >= 50:
        ans = 'C'
    elif m >= 40:
        ans = 'D'
    return ans

print('Physics grade: ', grade(p))
print('Chemistry grade: ', grade(c))
print('English grade: ', grade(e))
print('Maths grade: ', grade(m))
