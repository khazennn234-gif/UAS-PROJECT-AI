import cv2

# Membaca file gambar len.jpg yang berada dalam folder yang sama dengan file ReadImageData.py
img = cv2.imread("untad.jpg")

# Menampilkan data piksel citra
print('Image', img)

# Menampilkan data piksel citra
print('Image', img)

# Menampilkan Citra
cv2.imshow("Gambar Kampus Untad", img)

cv2.waitKey(0)
cv2.destroyAllWindows()