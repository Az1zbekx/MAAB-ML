# decision_tree_theory.py

"""
1. What is a Decision Tree, and how does it make predictions?
A Decision Tree — bu supervised learning modeli bo‘lib,
ma’lumotlarni feature bo‘yicha bo‘lib boradi (split).
Predict qilish: rootdan leafgacha yo‘l bosib,
leaf node dagi class/value ni qaytaradi.
"""

"""
2. Pure vs Impure node
Pure node: faqat bitta class bor (100% bir xil).
Impure node: bir nechta class aralash.
"""

"""
3. Role of Entropy and Gini impurity
Entropy va Gini node “qanchalik aralashganini” o‘lchaydi.
Past qiymat = toza (pure).
Split sifatini baholash uchun ishlatiladi.
"""

"""
4. Information Gain
Information Gain = splitdan keyingi impurity kamayishi.
Eng yaxshi splitni tanlash uchun ishlatiladi.
"""

"""
5. Why greedy algorithms?
Decision Tree har qadamda eng yaxshi splitni tanlaydi
(global optimalni emas, local optimalni).
Shuning uchun greedy deyiladi.
"""

"""
6. Why deep trees overfit?
Deep tree training data ni yodlab oladi (noise ham),
shuning uchun yangi data da yomon ishlaydi.
"""

"""
7. Bias–Variance tradeoff
Shallow tree: high bias, low variance (underfit).
Deep tree: low bias, high variance (overfit).
"""

"""
8. What is pruning?
Pruning — keraksiz branchlarni kesib tashlash.
Overfittingni kamaytiradi va modelni soddalashtiradi.
"""

"""
9. Why no feature scaling?
Decision Tree threshold va tartibga qaraydi,
distance ga emas, shuning uchun scaling kerak emas.
"""

"""
10. Advantage & Limitation
Advantage: tushunish va interpret qilish oson.
Limitation: overfittingga moyil.
"""


# Optional: simple print to verify file runs
if __name__ == "__main__":
    print("Decision Tree theory answers loaded successfully.")