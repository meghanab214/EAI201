def grade_from_marks(score):
    if score >= 90:
        return "A+"
    if score >= 85:
        return "A"
    if score >= 75:
        return "B+"
    if score >= 65:
        return "B"
    if score >= 50:
        return "C+"
    if score >= 40:
        return "C"
    return "Fail"

num_students = int(input("How many students? "))

for s in range(num_students):
    print("\n Student", s+1, "Details")
    name = input("Name: ")
    n_subjects = int(input("Number of subjects: "))

    subjects = []
    marks_list = []
    
    for k in range(n_subjects):
        subj = input(f"Subject {k+1} name: ")
        mark = int(input(f"Marks in {subj}: "))
        subjects.append(subj)
        marks_list.append(mark)

    total_marks = sum(marks_list)
    percentage = total_marks / n_subjects

    # Report card printing
    print("\n Report Card")
    print("Student:", name)
    for subj, mark in zip(subjects, marks_list):
        print(f"{subj:<12} Marks: {mark:3} | Grade: {grade_from_marks(mark)}")
    print(f"Total Percentage: {percentage:.2f}%")
    print(f"Overall Grade  : {grade_from_marks(percentage)}")
  
