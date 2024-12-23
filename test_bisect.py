import bisect

class Student:
    def __init__(self, name, age):
        self.name = name
        self.age = age

    def __repr__(self):
        return f"{self.name}: {self.age}"

# 创建一个学生列表并根据年龄排序
students = [Student("Alice", 22), Student("Bob", 20), Student("Charlie", 24)]
students.sort(key=lambda x: x.age)

# 定义一个函数来找到小于给定年龄的最大值
def find_max_less_than_t(students, t):
    # 使用相同的key参数来确保比较的一致性
    index = bisect.bisect_left(students, t, key=lambda x: x.age)
    if index == 0:
        return -1
    else:
        return students[index - 1]

# 使用示例
t = 25
result = find_max_less_than_t(students, 20)
print(result)

# std = Student("Wyo", 22)
# bisect.insort_right(students, std, key=lambda x: x.age)

# print(students)
# if result != -1:
#     print(f"小于{t}的最大年龄是{result}")
# else:
#     print(f"没有年龄小于{t}的学生")