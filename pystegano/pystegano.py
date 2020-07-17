import numpy as np
import pandas as pd
from skimage.transform import rescale
import skimage
import bitarray
import math
import cv2
import random
from scipy import fftpack

class lsb:
	"""
	Steganographic tools based on least significant bit (lsb).
	"""

	def __messageToBits(message):
		tag = "{:<10}".format(str(len(message)*8))
		message = tag+message
		code = bitarray.bitarray()
		code.frombytes(message.encode('utf-8'))
		code = "".join(['1' if x == True else '0' for x in code.tolist()])
		return code

	def __hideMessage(img, message):
		shape = img.shape
		img = img.flatten()
		code = list(message)
		code_len = len(code)
		for i,x in enumerate(img):
			if i*2 <code_len:
				zbits = list('{0:08b}'.format(x))[:6]+code[i*2:i*2+2]
				img[i] = int("".join(zbits), 2)
			else:
				return img.reshape(shape)
		return img.reshape(shape)

	def encode(cover_image, secret_message):
		"""
		Stores a secret_message (string) into a cover_image (image) using lsb steganography.
		"""

		image_height  = cover_image.shape[0]
		image_width   = cover_image.shape[1]
		image_chanels = cover_image.shape[2]

		image_size_bytes = image_height * image_width * image_chanels
		image_size_bits  = image_size_bytes * 8
		image_size_kbytes = image_size_bytes / 1024

		hide_size_bits  = image_size_bytes * 2
		hide_size_kbytes = (hide_size_bits / 8) / 1024

		message_size_kbytes = len(secret_message)/8000

		if message_size_kbytes > hide_size_kbytes:
			raise Exception('Message too big!')
		
		secret_message_bits = lsb.__messageToBits(secret_message)

		image_with_message = lsb.__hideMessage(cover_image, secret_message_bits)

		return image_with_message

	def decode(image):
		"""
		Retrives a secret_message (string) from an image (image) using lsb steganography.
		"""

		bit_message = ""
		bit_count = 0
		bit_length = 200
		for i,x in enumerate(image):
			for j,y in enumerate(x):
				for k,z in enumerate(y):
					zbits = '{0:08b}'.format(z)
					bit_message += zbits[-2:]
					bit_count += 2
					if bit_count == 80:
						try:
							decoded_tag = bitarray.bitarray(bit_message).tobytes().decode('utf-8')
							bit_length = int(decoded_tag)+80
							bit_message = ""
						except:
							raise Exception("Can't decode secret on this image!")
					elif bit_count >= bit_length:
						return bitarray.bitarray(bit_message).tobytes().decode('utf-8')

class lsb_grayscale:
	"""
	Steganographic tools based on least significant bit (lsb).
	Thesse methods receive an RGB image and outputs a grayscale image.
	"""

	def __messageToBits(message):
		tag = "{:<10}".format(str(len(message)*8))
		message = tag+message
		code = bitarray.bitarray()
		code.frombytes(message.encode('utf-8'))
		code = "".join(['1' if x == True else '0' for x in code.tolist()])
		return code

	def __hideMessage(img, message):
		shape = img.shape
		img = img.flatten()
		code = list(message)
		code_len = len(code)
		for i,x in enumerate(img):
			if i*2 <code_len:
				zbits = list('{0:08b}'.format(x))[:6]+code[i*2:i*2+2]
				img[i] = int("".join(zbits), 2)
			else:
				return img.reshape(shape)
		return img.reshape(shape)

	def __toGrayscale(imageA):
	    r = imageA[:,:,0]
	    g = imageA[:,:,1]
	    b = imageA[:,:,2]

	    shapeImage = imageA.shape;
	    retImage = np.zeros((shapeImage[0], shapeImage[1]))

	    for i in range(shapeImage[0]):
	        for j in range(shapeImage[1]):
	            retImage[i][j] = int(0.2126*r[i][j] + 0.7152*g[i][j] + 0.0722*b[i][j])

	    return retImage.astype("uint8")

	def encode(cover_image, secret_message):
		"""
		Stores a secret_message (string) into a cover_image (image) using lsb steganography.
		This method receive an RGB image and outputs a grayscale image.
		"""

		cover_image = lsb_grayscale.__toGrayscale(cover_image)

		image_height  = cover_image.shape[0]
		image_width   = cover_image.shape[1]

		image_size_bytes = image_height * image_width
		image_size_bits  = image_size_bytes * 8
		image_size_kbytes = image_size_bytes / 1024

		hide_size_bits  = image_size_bytes * 2
		hide_size_kbytes = (hide_size_bits / 8) / 1024

		message_size_kbytes = len(secret_message)/8000

		if message_size_kbytes > hide_size_kbytes:
			raise Exception('Message too big!')
		
		secret_message_bits = lsb_grayscale.__messageToBits(secret_message)

		image_with_message = lsb_grayscale.__hideMessage(cover_image, secret_message_bits)

		return image_with_message

	def decode(cover_image):
		"""
		Retrives a secret_message (string) from an image (image) using lsb steganography.
		This method receive an RGB image and outputs a grayscale image.
		"""

		bit_message = ""
		bit_count = 0
		bit_length = 200
		for i,x in enumerate(cover_image):
			for j,y in enumerate(x):
				zbits = '{0:08b}'.format(y)
				bit_message += zbits[-2:]
				bit_count += 2
				if bit_count == 80:
					try:
						decoded_tag = bitarray.bitarray(bit_message).tobytes().decode('utf-8')
						bit_length = int(decoded_tag)+80
						bit_message = ""
					except:
						raise Exception("Can't decode secret on this image!")
				elif bit_count >= bit_length:
					return bitarray.bitarray(bit_message).tobytes().decode('utf-8')

class ssb4_grayscale:
	"""
	Steganographic tools based on fourth significant bit (lsb).
	Thesse methods receive an RGB image and outputs a grayscale image.
	"""

	def __messageToBits(message):
		tag = "{:<10}".format(str(len(message)*8))
		message = tag+message
		code = bitarray.bitarray()
		code.frombytes(message.encode('utf-8'))
		code = "".join(['1' if x == True else '0' for x in code.tolist()])
		return code

	def __hideMessage(img, message):
	    vetNormalize = [232, 233, 234, 235, 236, 237, 238, 239, 248, 249, 250, 251, 252, 253, 254, 255]

	    shape = img.shape
	    img = img.flatten()
	    code = list(message)
	    code_len = len(code)
	    for i,x in enumerate(img):
	        if i <code_len:
	            valorPixel = x
	            tempDiff = 256
	            zListTemp = list('{0:08b}'.format(x))
	            zbits = zListTemp[:4]+[code[i]]+zListTemp[5:]
	            img[i] = int("".join(zbits), 2)

	            for j in vetNormalize:
	                if abs((img[i] & j)-img[i]) < tempDiff:
	                    valorPixel = (img[i] & j)
	                    tempDiff = abs((img[i] & j)-img[i])

	            img[i] = valorPixel
	        else:
	            return img.reshape(shape)
	    return img.reshape(shape)

	def __toGrayscale(imageA):
	    r = imageA[:,:,0]
	    g = imageA[:,:,1]
	    b = imageA[:,:,2]

	    shapeImage = imageA.shape;
	    retImage = np.zeros((shapeImage[0], shapeImage[1]))

	    for i in range(shapeImage[0]):
	        for j in range(shapeImage[1]):
	            retImage[i][j] = int(0.2126*r[i][j] + 0.7152*g[i][j] + 0.0722*b[i][j])

	    return retImage.astype("uint8")

	def encode(cover_image, secret_message):
		"""
		Stores a secret_message (string) into a cover_image (image) using ssb-4 steganography.
		This method receive an RGB image and outputs a grayscale image.
		"""

		cover_image = ssb4_grayscale.__toGrayscale(cover_image)

		image_height  = cover_image.shape[0]
		image_width   = cover_image.shape[1]

		image_size_bytes = image_height * image_width
		image_size_bits  = image_size_bytes * 8
		image_size_kbytes = image_size_bytes / 1024

		hide_size_bits  = image_size_bytes
		hide_size_kbytes = (hide_size_bits / 8) / 1024

		message_size_kbytes = len(secret_message)/8000

		if message_size_kbytes > hide_size_kbytes:
			raise Exception('Message too big!')
		
		secret_message_bits = ssb4_grayscale.__messageToBits(secret_message)

		image_with_message = ssb4_grayscale.__hideMessage(cover_image, secret_message_bits)

		return image_with_message

	def decode(cover_image):
		"""
		Retrives a secret_message (string) from an image (image) using ssb-4 steganography.
		This method receive an RGB image and outputs a grayscale image.
		"""

		bit_message = ""
		bit_count = 0
		bit_length = 200
		for i,x in enumerate(cover_image):
			for j,y in enumerate(x):
				zbits = '{0:08b}'.format(y)
				bit_message += zbits[4]
				bit_count += 1
				if bit_count == 80:
					try:
						decoded_tag = bitarray.bitarray(bit_message).tobytes().decode('utf-8')
						bit_length = int(decoded_tag)+80
						bit_message = ""
					except:
						print("Image does not have decode tag. Image is either not encoded or, at least, not encoded in a way this decoder recognizes")
						return
				elif bit_count >= bit_length:
					return bitarray.bitarray(bit_message).tobytes().decode('utf-8')

class ssbn_grayscale:
	"""
	Steganographic tools based on random significant bit (lsb).
	Thesse methods receive an RGB image and outputs a grayscale image.
	"""

	def __messageToBits(message):
		tag = "{:<10}".format(str(len(message)*8))
		message = tag+message
		code = bitarray.bitarray()
		code.frombytes(message.encode('utf-8'))
		code = "".join(['1' if x == True else '0' for x in code.tolist()])
		return code

	def __hideMessage(img, message):
	    vetNormalize1 = [225, 227, 229, 231, 233, 235, 237, 239, 241, 243, 245, 247, 249, 251, 253, 255]
	    vetNormalize2 = [226, 227, 230, 231, 234, 235, 238, 239, 242, 243, 246, 247, 250, 251, 254, 255]
	    vetNormalize3 = [228, 229, 230, 231, 236, 237, 238, 239, 244, 245, 246, 247, 252, 253, 254, 255]
	    vetNormalize4 = [232, 233, 234, 235, 236, 237, 238, 239, 248, 249, 250, 251, 252, 253, 254, 255]

	    shape = img.shape
	    img = img.flatten()
	    code = list(message)
	    code_len = len(code)
	    password = [random.randint(0, 3) for i in range(code_len)]

	    for i,x in enumerate(img):
	        if i <code_len:
	            valorPixel = x
	            tempDiff = 256
	            zListTemp = list('{0:08b}'.format(x))

	            if password[i] == 0:
	                zbits = zListTemp[:7]+[code[i]]
	                img[i] = int("".join(zbits), 2)

	                for j in vetNormalize1:
	                    if abs((img[i] & j)-img[i]) < tempDiff:
	                        valorPixel = (img[i] & j)
	                        tempDiff = abs((img[i] & j)-img[i])
	            elif password[i] == 1:
	                zbits = zListTemp[:6]+[code[i]]+zListTemp[7:]
	                img[i] = int("".join(zbits), 2)

	                for j in vetNormalize2:
	                    if abs((img[i] & j)-img[i]) < tempDiff:
	                        valorPixel = (img[i] & j)
	                        tempDiff = abs((img[i] & j)-img[i])
	            elif password[i] == 2:
	                zbits = zListTemp[:5]+[code[i]]+zListTemp[6:]
	                img[i] = int("".join(zbits), 2)

	                for j in vetNormalize3:
	                    if abs((img[i] & j)-img[i]) < tempDiff:
	                        valorPixel = (img[i] & j)
	                        tempDiff = abs((img[i] & j)-img[i])
	            elif password[i] == 3:
	                zbits = zListTemp[:4]+[code[i]]+zListTemp[5:]
	                img[i] = int("".join(zbits), 2)

	                for j in vetNormalize4:
	                    if abs((img[i] & j)-img[i]) < tempDiff:
	                        valorPixel = (img[i] & j)
	                        tempDiff = abs((img[i] & j)-img[i])

	            img[i] = valorPixel
	        else:
	            return img.reshape(shape), password
	    return img.reshape(shape), password

	def __toGrayscale(imageA):
	    r = imageA[:,:,0]
	    g = imageA[:,:,1]
	    b = imageA[:,:,2]

	    shapeImage = imageA.shape;
	    retImage = np.zeros((shapeImage[0], shapeImage[1]))

	    for i in range(shapeImage[0]):
	        for j in range(shapeImage[1]):
	            retImage[i][j] = int(0.2126*r[i][j] + 0.7152*g[i][j] + 0.0722*b[i][j])

	    return retImage.astype("uint8")

	def encode(cover_image, secret_message):
		"""
		Stores a secret_message (string) into a cover_image (image) using ssb-n steganography.
		This method receive an RGB image and outputs a grayscale image and the password.
		"""

		cover_image = ssbn_grayscale.__toGrayscale(cover_image)

		image_height  = cover_image.shape[0]
		image_width   = cover_image.shape[1]

		image_size_bytes = image_height * image_width
		image_size_bits  = image_size_bytes * 8
		image_size_kbytes = image_size_bytes / 1024

		hide_size_bits  = image_size_bytes
		hide_size_kbytes = (hide_size_bits / 8) / 1024

		message_size_kbytes = len(secret_message)/8000

		if message_size_kbytes > hide_size_kbytes:
			raise Exception('Message too big!')
		
		secret_message_bits = ssbn_grayscale.__messageToBits(secret_message)

		image_with_message, password = ssbn_grayscale.__hideMessage(cover_image, secret_message_bits)

		return image_with_message, password

	def decode(cover_image, password):
		"""
		Retrives a secret_message (string) from an image (image) using ssb-n steganography.
		This method receive an RGB image and outputs a grayscale image.
		"""

		bit_message = ""
		bit_count = 0
		bit_length = 200
		password_counter = 0
		for i,x in enumerate(cover_image):
			for j,y in enumerate(x):
				zbits = '{0:08b}'.format(y)

				if password[password_counter] == 0:
					bit_message += zbits[7]
				elif password[password_counter] == 1:
					bit_message += zbits[6]
				elif password[password_counter] == 2:
					bit_message += zbits[5]
				elif password[password_counter] == 3:
					bit_message += zbits[4]

				password_counter += 1
				bit_count += 1
				if bit_count == 80:
					try:
						decoded_tag = bitarray.bitarray(bit_message).tobytes().decode('utf-8')
						bit_length = int(decoded_tag)+80
						bit_message = ""
					except:
						print("Image does not have decode tag. Image is either not encoded or, at least, not encoded in a way this decoder recognizes")
						return
				elif bit_count >= bit_length:
					return bitarray.bitarray(bit_message).tobytes().decode('utf-8')

class dct:
	"""
	Steganographic tools based on discrete cosine transform (dct).
	"""

	def __imagemParaBits(imagemSecreta):
		bitsSecretos = []

		for valorAtual in imagemSecreta:
			bits = format(valorAtual, '08b')
			for bit in bits:
				bitsSecretos.append(bit)

		return bitsSecretos

	def __bitsParaImagem(bitsSecretos):
		shape = []
		valoresImagem = []

		nBits = 1

		temp = ""

		for bit in bitsSecretos:
			temp += bit

			if nBits % 8 == 0:
				valoresImagem.append(int(temp, 2))
				temp = ""

			nBits += 1

		return valoresImagem

	def encode(cover_image, secret_bits, channel):
		"""
		Stores a secret_message (string) into a cover_image (image) using dct steganography.

		Channel is an integer representing which RGB channel will store the secret_bits.

		+---------+-----------+
		|  Value  |  Channel  |
		+---------+-----------+
		|  0      |  Blue     |
		|  1      |  Greeen   |
		|  2      |  Red      |
		+---------+-----------+
		"""

		bImg,gImg,rImg = cv2.split(cover_image)

		if channel == 0:
			im = bImg
		elif channel == 1:
			im = gImg
		elif channel == 2:
			im = rImg

		row, col = im.shape[:2]

		blocos = []

		for i in range(0, row, 8):
			for j in range(0, col, 8):
				b = im[i:i+8,j:j+8]
				blocos.append(b)

		blocosDct = [np.round(fftpack.dct(b)) for b in blocos]

		bitMess = dct.__imagemParaBits(secret_bits)

		numBits = len(bitMess)

		count = 0

		arquivoTexto = ""

		for B in blocosDct:
			C = np.array(B)

			DC = C[0][0]

			if bitMess[count] == "0":
				DC = 4096
				arquivoTexto += "0"
			else:
				DC = 0
				arquivoTexto += "1"

			count += 1

			for i in range(8):
				for j in range(8):
					B[i][j] = C[i][j]

			if count < numBits:
				B[0][0] = DC
			else:
				break;

		if count < numBits:
			raise Exception('Message too big!')

		blocosIdct = [fftpack.idct(b) for b in blocosDct]

		im1 = [[0 for i in range(col)] for j in range(row)]

		noBloco = 0

		for i1 in range(0, row, 8):
			for j1 in range(0, col, 8):
				for i2 in range(0, 8):
					for j2 in range(0, 8):
						im1[i1+i2][j1+j2] = blocosIdct[noBloco][i2][j2]
				noBloco += 1

		im1 = np.array(im1)

		im1 = im1 + (-1*im1.min())

		im1 = im1 * 255 / im1.max()

		im1 = np.uint8(im1)

		if channel == 0:
			im1 = cv2.merge((im1,gImg,rImg))
		elif channel == 1:
			im1 = cv2.merge((bImg,im1,rImg))
		elif channel == 2:
			im1 = cv2.merge((bImg,gImg,im1))

		return im1

	def decode(cover_image, len_secret_bytes, channel):
		"""
		Retrives a secret_message (string) from an image (image) using dct steganography.

		Channel is an integer representing in which RGB channel were stored the secret_bits.

		+---------+-----------+
		|  Value  |  Channel  |
		+---------+-----------+
		|  0      |  Blue     |
		|  1      |  Greeen   |
		|  2      |  Red      |
		+---------+-----------+
		"""

		bImg,gImg,rImg = cv2.split(cover_image)

		if channel == 0:
			im = bImg
		elif channel == 1:
			im = gImg
		elif channel == 2:
			im = rImg

		row, col = im.shape[:2]

		blocos = []

		for i in range(0, row, 8):
			for j in range(0, col, 8):
				b = im[i:i+8,j:j+8]
				blocos.append(b)

		blocosDct = [np.round(fftpack.dct(b)) for b in blocos]

		bitMess = []

		count = 0

		arquivoTexto = ""

		for B in blocosDct:
			C = np.array(B)

			DC = C[0][0]

			if DC > 2100:
				bitMess.append("0")
				arquivoTexto += "0"
			else:
				bitMess.append("1")
				arquivoTexto += "1"
			
			count += 1

			if count == len_secret_bytes:
				break

		imagemSecreta = dct.__bitsParaImagem(bitMess)

		return imagemSecreta

class fft:
	"""
	Steganographic tools based on fast fourier transform (fft).
	"""

	def __imagemParaBits(imagemSecreta):
		bitsSecretos = []

		for valorAtual in imagemSecreta:
			bits = format(valorAtual, '08b')
			for bit in bits:
				bitsSecretos.append(bit)

		return bitsSecretos

	def __bitsParaImagem(bitsSecretos):
		shape = []
		valoresImagem = []

		nBits = 1

		temp = ""

		for bit in bitsSecretos:
			temp += bit

			if nBits % 8 == 0:
				valoresImagem.append(int(temp, 2))
				#print(int(temp, 2))
				temp = ""

			nBits += 1

		return valoresImagem

	def encode(cover_image, secret_bits, channel):
		"""
		Stores a secret_message (string) into a cover_image (image) using fft steganography.

		Channel is an integer representing which RGB channel will store the secret_bits.

		+---------+-----------+
		|  Value  |  Channel  |
		+---------+-----------+
		|  0      |  Blue     |
		|  1      |  Greeen   |
		|  2      |  Red      |
		+---------+-----------+
		"""

		bImg,gImg,rImg = cv2.split(cover_image)

		if channel == 0:
			im = bImg
		elif channel == 1:
			im = gImg
		elif channel == 2:
			im = rImg

		row, col = im.shape[:2]

		blocos = []

		for i in range(0, row, 8):
			for j in range(0, col, 8):
				b = im[i:i+8,j:j+8]
				blocos.append(b)

		blocosDct = [np.round(fftpack.fft(b)) for b in blocos]

		bitMess = fft.__imagemParaBits(secret_bits)

		numBits = len(bitMess)

		count = 0

		arquivoTexto = ""

		for B in blocosDct:
			C = np.array(B)

			DC = C[0][0]

			if bitMess[count] == "0":
				DC = 2048
				arquivoTexto += "0"
			else:
				DC = 0
				arquivoTexto += "1"

			count += 1

			for i in range(8):
				for j in range(8):
					B[i][j] = C[i][j]

			if count < numBits:
				B[0][0] = DC
			else:
				break;

		if count < numBits:
			raise Exception('Message too big!')

		blocosIdct = [fftpack.ifft(b) for b in blocosDct]

		im1 = [[0 for i in range(col)] for j in range(row)]

		noBloco = 0

		for i1 in range(0, row, 8):
			for j1 in range(0, col, 8):
				for i2 in range(0, 8):
					for j2 in range(0, 8):
						im1[i1+i2][j1+j2] = blocosIdct[noBloco][i2][j2]
				noBloco += 1

		im1 = np.array(im1)

		im1 = im1 + (-1*im1.min())

		im1 = im1 * 255 / im1.max()

		im1 = np.uint8(im1)

		if channel == 0:
			im1 = cv2.merge((im1,gImg,rImg))
		elif channel == 1:
			im1 = cv2.merge((bImg,im1,rImg))
		elif channel == 2:
			im1 = cv2.merge((bImg,gImg,im1))

		return im1

	def decode(cover_image, len_secret_bytes, channel):
		"""
		Retrives a secret_message (string) from an image (image) using fft steganography.

		Channel is an integer representing in which RGB channel were stored the secret_bits.

		+---------+-----------+
		|  Value  |  Channel  |
		+---------+-----------+
		|  0      |  Blue     |
		|  1      |  Greeen   |
		|  2      |  Red      |
		+---------+-----------+
		"""

		bImg,gImg,rImg = cv2.split(cover_image)

		if channel == 0:
			im = bImg
		elif channel == 1:
			im = gImg
		elif channel == 2:
			im = rImg

		row, col = im.shape[:2]

		blocos = []

		for i in range(0, row, 8):
			for j in range(0, col, 8):
				b = im[i:i+8,j:j+8]
				blocos.append(b)

		blocosDct = [np.round(fftpack.fft(b)) for b in blocos]

		bitMess = []

		count = 0

		arquivoTexto = ""

		for B in blocosDct:
			C = np.array(B)

			DC = C[0][0]

			if DC > 2048:
				bitMess.append("0")
				arquivoTexto += "0"
			else:
				bitMess.append("1")
				arquivoTexto += "1"
			
			count += 1

			if count == len_secret_bytes:
				break

		imagemSecreta = fft.__bitsParaImagem(bitMess)

		return imagemSecreta
