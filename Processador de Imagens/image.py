import numpy as np
from PIL import Image


def open_image(file_path):
    image = Image.open(file_path)
    return image


##Manipulaçção


# Aula 2
def negative(img: Image.Image):
    arr = np.array(img)
    response = arr[:] = 255 - arr
    return response


def tLog(c, img: Image.Image):
    arr = np.array(img)
    arr = arr / 255.0

    log_image = c * np.log2(1 + arr)

    log_image[log_image < 0] = 0
    log_image[log_image > 1] = 1

    return (log_image * 255).astype(np.uint8)


def gammaCorrection(c, x, img: Image.Image):
    arr = np.array(img)
    arr = arr / 255.0

    gamma_image = c * (arr**x)

    gamma_image[gamma_image < 0] = 0
    gamma_image[gamma_image > 1] = 1

    return (gamma_image * 255).astype(np.uint8)


def linear(imagem, dic: dict):
    nova_imagem = imagem.copy()
    for intensidade in range(256):
        nova_imagem[nova_imagem == intensidade] = dic[
            intensidade
        ]  # nova intensidade definida na funcao

    return nova_imagem


def string_para_bits(texto):
    bits = ""
    for char in texto:
        ascii_valor = ord(char)  # valor ASCII
        binario = bin(ascii_valor)[2:]  # ASCII para binario e removendo "0b"
        binario = binario.zfill(8)  # 8 bits
        bits += binario
    return bits


def esteganografia(image: Image, mensagem: str):
    arr = np.array(image)
    mensagem = string_para_bits(mensagem) + "1111111111111110"  # 6553

    if len(mensagem) > arr.size:
        raise ValueError(
            "A imagem não comporta o tamanho da mensagem a ser esteganografada."
        )
    msg_bits = []
    for bit in mensagem:
        msg_bits.append(int(bit))

    lins, cols = arr.shape
    for i in range(lins):
        for j in range(cols):
            if len(msg_bits) > 0:
                arr[i][j] = (arr[i][j] & 254) | msg_bits[
                    0
                ]  # (r AND 11111110) OR bit_substitui
                msg_bits = msg_bits[1:]  # descarta o bit que ja foi colocado na imagem

    return arr


def bits_para_string(bits):
    texto = ""
    for i in range(0, len(bits), 8):  # divide os bits em grupos de 8
        byte = bits[i : i + 8]
        valor_decimal = int(byte, 2)  # byte binário o valor decimal
        texto += chr(valor_decimal)  # decimal para string
    return texto


def decodificar(image: Image):
    arr = np.array(image)
    lins, cols = arr.shape
    mensagem = ""
    for i in range(lins):
        for j in range(cols):
            mensagem += str(arr[i][j] & 1)

    fim_mensagem = mensagem.find("1111111111111110")
    msg = mensagem[:fim_mensagem]
    return bits_para_string(msg)


# Aula 3
def histograma(image: Image.Image):
    arr = np.array(image)
    hist = {}
    for i in range(256):
        hist[i] = np.count_nonzero(arr == i)
    return hist


def hist_normalizado(arr):
    hist = {}
    lins, cols = arr.shape
    for i in range(256):
        hist[i] = np.count_nonzero(arr == i) / (lins * cols)
    return hist


def prrk(j, hist):
    pr = 0
    for i in range(j + 1):
        pr = pr + hist[i]  # (np.count_nonzero(arr == i) / (m * n))
    return pr


def hist_equalizado(image: Image):
    arr = np.array(image)
    lins, cols = arr.shape
    aux = np.empty((lins, cols))
    hist = hist_normalizado(arr)
    for i in range(lins):
        for j in range(cols):
            aux[i][j] = 255 * prrk(arr[i][j], hist)
    return aux.astype(np.uint8)


def alargamento_contraste(image: Image.Image):
    imagem: np.ndarray[int] = np.array(image)
    menor_pixel = imagem.min()
    maior_pixel = imagem.max()
    imagem[:] = ((imagem - menor_pixel) / (maior_pixel - menor_pixel)) * (255 - 0) + 0


def limiar(image: Image):
    imagem = np.array(image)
    alargamento_contraste(image)
    media = imagem.mean()
    imagem[imagem <= media] = 0
    imagem[imagem > media] = 255
    return imagem


# Aula 6
def convolucao(image: Image.Image, kernel):
    imagem: np.ndarray[int] = np.array(image)
    kernel = np.flipud(np.fliplr(kernel))  # rotacionando o kernel em 180 graus

    altura_imagem, largura_imagem = imagem.shape
    altura_kernel, largura_kernel = kernel.shape

    percorrer_linha = altura_imagem - altura_kernel + 1
    percorrer_coluna = largura_imagem - largura_kernel + 1

    linha_central = altura_kernel // 2
    coluna_central = largura_kernel // 2

    saida = np.copy(imagem)
    for i in range(percorrer_linha):
        for j in range(percorrer_coluna):
            janela = imagem[i : i + altura_kernel, j : j + largura_kernel]
            saida[i + linha_central, j + coluna_central] = np.sum(janela * kernel)

    return saida


def suavizacao_media(tamanho: int, image: Image.Image):
    imagem = np.array(image)
    kernel = np.full((tamanho, tamanho), 1 / (tamanho**2))
    saida = convolucao(imagem, kernel)
    return saida


def kernel_ponderado(tamanho: int):
    kernel = np.zeros((tamanho, tamanho))
    lins, cols = kernel.shape
    valor_centro = tamanho + 1

    for i in range(tamanho):
        for j in range(tamanho):
            distancia = abs(i - tamanho // 2) + abs(j - tamanho // 2)
            kernel[i, j] = valor_centro - distancia - 1
    kernel[lins // 2, cols // 2] = valor_centro
    return kernel


def suavizacao_ponderada(tamanho: int, image: Image.Image):
    imagem = np.array(image)
    kernel = kernel_ponderado(tamanho)
    kernel = kernel * (1 / kernel.sum())
    saida = convolucao(imagem, kernel)
    return saida


def suavizacao_mediana(tamanho_janela: int, image: Image.Image):
    imagem = np.array(image)
    altura_imagem, largura_imagem = imagem.shape

    percorrer_linha = altura_imagem - tamanho_janela + 1
    percorrer_coluna = largura_imagem - tamanho_janela + 1

    linha_central = tamanho_janela // 2
    coluna_central = tamanho_janela // 2

    saida = np.copy(imagem)
    for i in range(percorrer_linha):
        for j in range(percorrer_coluna):
            janela = imagem[i : i + tamanho_janela, j : j + tamanho_janela]
            saida[i + linha_central, j + coluna_central] = np.median(janela)

    return saida


# Aula 7
def imagem_laplaciana(image: Image.Image):
    imagem: np.ndarray[int] = np.array(image)
    kernel = np.array([[0, 1, 0], [1, -4, 1], [0, 1, 0]])
    # kernel = np.array([[-1, -1, -1], [-1, 8, -1], [-1, -1, -1]])

    altura_imagem, largura_imagem = imagem.shape
    altura_kernel, largura_kernel = kernel.shape

    percorrer_linha = altura_imagem - altura_kernel + 1
    percorrer_coluna = largura_imagem - largura_kernel + 1

    linha_central = altura_kernel // 2
    coluna_central = largura_kernel // 2

    saida = np.copy(imagem)
    for i in range(percorrer_linha):
        for j in range(percorrer_coluna):
            janela = imagem[i : i + altura_kernel, j : j + largura_kernel]
            saida[i + linha_central, j + coluna_central] = np.sum(janela * kernel)

    return saida


def agucamento_laplace(image: Image.Image):
    imagem = np.array(image)
    imagem = imagem / 255.0
    laplace = imagem_laplaciana(imagem)
    saida = imagem + (-1 * laplace)
    saida = ((saida - np.min(saida)) / (np.max(saida) - np.min(saida)) * 255).astype(
        np.uint8
    )
    return saida


def agucamento_highboost(k: int, image: Image.Image):
    imagem = np.array(image)
    imagem = imagem / 255.0
    borrada = suavizacao_media(3, imagem)
    mascara = imagem - borrada
    saida = imagem + (k * mascara)
    saida = ((saida - np.min(saida)) / (np.max(saida) - np.min(saida)) * 255).astype(
        np.uint8
    )
    return saida


# Aula 8
def sobel_x(imagem):
    sobel_x = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])
    filtro_x = convolucao(imagem, sobel_x)
    # filtro_x = filtro_x / 255
    return filtro_x


def sobel_y(imagem):
    sobel_y = np.array([[1, 2, 1], [0, 0, 0], [-1, -2, -1]])
    filtro_y = convolucao(imagem, sobel_y)
    # filtro_y = filtro_y / 255
    return filtro_y


def gradiente(image: Image.Image):
    imagem: np.ndarray[int] = np.array(image)
    imagem = imagem / 255.0
    gx = sobel_x(imagem)
    gy = sobel_y(imagem)
    gradiente = np.abs(gx) + np.abs(gy)
    # gradiente = np.sqrt((gx**2) + (gy**2))
    gradiente = (
        (gradiente - np.min(gradiente)) / (np.max(gradiente) - np.min(gradiente)) * 255
    ).astype(
        np.uint8
    )  # normalizacao

    return gradiente


# Aula 12
def dft(imagem: np.ndarray):
    m, n = imagem.shape
    transformada = np.zeros((m, n), dtype=complex)

    for u in range(m):
        for v in range(n):
            valor = 0
            for x in range(m):
                for y in range(n):
                    pixel = imagem[x, y]
                    valor += pixel * np.exp(-2j * np.pi * ((u * x / m) + (v * y / n)))
            transformada[u, v] = valor

    return transformada  # colocar em abs


def idft(imagem: np.ndarray):
    
    m, n = imagem.shape
    transformada = np.zeros((m, n), dtype=complex)

    for x in range(m):
        for y in range(n):
            valor = 0
            for u in range(m):
                for v in range(n):
                    pixel = imagem[u, v]
                    valor += pixel * np.exp(2j * np.pi * ((u * x / m) + (v * y / n)))
            transformada[x, y] = valor
    idft = transformada / (m * n)
    return idft


# Aula 15
def pixel_rgb_to_hsv(rgb):
    r, g, b = rgb / 255.0
    cmax = np.max(rgb)
    cmin = np.min(rgb)
    delta = cmax - cmin

    # Calculando o componente Hue (matiz)
    if delta == 0:
        h = 0
    elif cmax == r:
        h = 60 * ((g - b) / delta % 6)
    elif cmax == g:
        h = 60 * ((b - r) / delta + 2)
    else:
        h = 60 * ((r - g) / delta + 4)

    if h < 0:
        h = h + 360

    #    if delta == 0:
    #        h = 0
    #    elif cmax == r:
    #        h = (60 * ((g - b) / delta)) % 360
    #    elif cmax == g:
    #        h = (60 * ((b - r) / delta) + 120) % 360
    #    else:
    #        h = (60 * ((r - g) / delta) + 240) % 360

    if cmax == 0:
        s = 0
    else:
        s = (delta / cmax) * 100

    v = (cmax / 255) * 100

    return [h, s, v]


def imagem_rgb_to_hsv(rgb_image):
    hsv_image = np.apply_along_axis(pixel_rgb_to_hsv, -1, rgb_image)
    return hsv_image


def pixel_hsv_to_rgb(hsv):
    h, s, v = hsv
    c = s * v / 100
    x = c * (1 - abs((h / 60) % 2 - 1))
    m = v - c

    if 0 <= h < 60:
        r, g, b = c, x, 0
    elif 60 <= h < 120:
        r, g, b = x, c, 0
    elif 120 <= h < 180:
        r, g, b = 0, c, x
    elif 180 <= h < 240:
        r, g, b = 0, x, c
    elif 240 <= h < 300:
        r, g, b = x, 0, c
    else:
        r, g, b = c, 0, x

    r, g, b = (r + m) * 255, (g + m) * 255, (b + m) * 255
    return [int(r) / 100, int(g) / 100, int(b) / 100]


def imagem_hsv_to_rgb(hsv_image):
    rgb_image = np.apply_along_axis(pixel_hsv_to_rgb, -1, hsv_image)
    return rgb_image.astype(np.uint8)


def pixel_media_rgb(rgb):
    media = np.sum(rgb) / 3
    return media


def imagem_cinza_media(imagem_rgb):
    imagem_cinza = np.apply_along_axis(pixel_media_rgb, -1, imagem_rgb)
    return imagem_cinza.astype(np.uint8)


def pixel_media_ponderada(pixel):
    R, G, B = pixel
    ponderada = R * 0.3 + G * 0.56 + B * 0.11
    return ponderada.astype(np.uint8)


def imagem_cinza_ponderada(imagem_rgb):
    imagem_cinza = np.apply_along_axis(pixel_media_ponderada, -1, imagem_rgb)
    return imagem_cinza


# Negativo RGB pode usar a funcao ja implementada


# # Aula 16
def chromakey(image, back, dist=50):
    img = np.array(image.resize((200, 200)))
    m, n , c = img.shape
    back_ground = np.array(back.resize((m + 1, n + 1)))
    
    if (m > back_ground.shape[0]) or (n > back_ground.shape[1]):
        print("Imagem de fundo é menor que imagem original")
        return img
    
    nova_imagem = img.copy()
    for i in range(m):
        for j in range(n):
            pixel = img[i,j,:]

            d = np.sqrt(((0 - pixel[0]) ** 2) + ((255 - pixel[1]) ** 2) + ((0 - pixel[2]) ** 2))
            if d <= dist:
                nova_imagem[i,j,:] = back_ground[i,j,:]

    return nova_imagem


def convolucao_rgb(image: Image, kernel):
    imagem = np.array(image)
    saida = imagem.copy()
    for i in range(3):
        saida[:, :, i] = convolucao(Image.fromarray(imagem[:, :, i]), kernel)

    return saida


def suavizacao_media_rgb(tamanho, image: Image):
    imagem = np.array(image)
    saida = imagem.copy()
    for i in range(3):
        saida[:, :, i] = suavizacao_media(tamanho, Image.fromarray(imagem[:, :, i]))
    return saida


def suavizacao_ponderada_rgb(tamanho: int, image: Image):
    imagem = np.array(image)
    saida = imagem.copy()
    for i in range(3):
        saida[:, :, i] = suavizacao_ponderada(
            tamanho, Image.fromarray(imagem[:, :, i])
        )  # converter saida para imagem
    return saida


def suavizacao_mediana_rgb(tamanho: int, image: Image):
    imagem = np.array(image)
    saida = imagem.copy()
    for i in range(3):
        saida[:, :, i] = suavizacao_mediana(tamanho, imagem[:, :, i])

    return saida


def agucamento_laplace_rgb(image: Image):
    imagem = np.array(image)
    saida = imagem.copy()
    for i in range(3):
        saida[:, :, i] = agucamento_laplace(Image.fromarray(imagem[:, :, i]))

    return saida


def agucamento_highboost_rgb(k, image: Image):
    imagem = np.array(image)
    saida = imagem.copy()
    for i in range(3):
        saida[:, :, i] = agucamento_highboost(k, Image.fromarray(imagem[:, :, i]))

    return saida


def gradiente_rgb(image: Image):
    imagem = np.array(image)
    saida = imagem.copy()
    for i in range(3):
        saida[:, :, i] = gradiente(Image.fromarray(imagem[:, :, i]))

    return saida


def histograma_RGBI(image: Image):
    imagem = np.array(image)
    hists = []
    for i in range(3):
        hist = histograma(Image.fromarray(imagem[:, :, i]))
        hists.append(hist)

    gray = imagem_cinza_ponderada(image)
    hist_I = histograma(gray)
    hists.append(hist_I)
    return hists


def hist_equalizado_rgb(image: Image):
    imagem = np.array(image)
    saida = imagem.copy()
    for i in range(3):
        saida[:, :, i] = hist_equalizado(Image.fromarray(imagem[:, :, i]))

    return saida


def dft_rgb(imagem: np.ndarray):
    saida = imagem.copy()
    for i in range(3):
        saida[:, :, i] = abs(dft(imagem[:, :, i]))

    return saida


def idft_rgb(imagem: np.ndarray):
    saida = imagem.copy()
    for i in range(3):
        saida[:, :, i] = abs(idft(imagem[:, :, i]))

    return saida


def pixel_sepia(pixel):
    r, g, b = pixel

    R = min(255, int(r * 0.393 + g * 0.769 + b * 0.189))
    G = min(255, int(r * 0.349 + g * 0.686 + b * 0.168))
    B = min(255, int(r * 0.272 + g * 0.534 + b * 0.131))

    return np.array([R, G, B])


def imagem_sepia(imagem):
    sepia = np.apply_along_axis(pixel_sepia, -1, imagem)
    return sepia.astype(np.uint8)


# Aula 17
def escala_vizinho_proximo(imagem, escala_x, escala_y):
    imagem = np.array(imagem)
    altura, largura, canais = imagem.shape
    nova_altura = int(altura * escala_y)
    nova_largura = int(largura * escala_x)

    nova_imagem = np.zeros((nova_altura, nova_largura, canais), dtype=np.uint8)

    for y in range(nova_altura):
        for x in range(nova_largura):
            x_original = int(x / escala_x)
            y_original = int(y / escala_y)
            nova_imagem[y, x] = imagem[y_original, x_original]

    return nova_imagem


def rotacao_vizinho_proximo(imagem, angulo):  # angulo em radianos
    imagem = np.array(imagem)
    m, n, canais = imagem.shape
    centro_x = n / 2
    centro_y = m / 2

    nova_imagem = np.zeros_like(imagem)

    for y in range(m):
        for x in range(n):
            novo_x = int(
                (x - centro_x) * np.cos(angulo)
                - (y - centro_y) * np.sin(angulo)
                + centro_x
            )
            novo_y = int(
                (x - centro_x) * np.sin(angulo)
                + (y - centro_y) * np.cos(angulo)
                + centro_y
            )

            if 0 <= novo_x < n and 0 <= novo_y < m:
                nova_imagem[y, x] = imagem[novo_y, novo_x]

    return nova_imagem


def escala_linear(imagem, escala_x, escala_y):
    imagem = np.array(imagem)
    altura, largura, canais = imagem.shape
    nova_altura = int(altura * escala_y)
    nova_largura = int(largura * escala_x)

    nova_imagem = np.zeros((nova_altura, nova_largura, canais), dtype=np.uint8)

    for y in range(nova_altura):
        for x in range(nova_largura):
            x_original = x / escala_x
            y_original = y / escala_y

            x1, y1 = int(x_original), int(y_original)
            x2, y2 = min(x1 + 1, largura - 1), min(y1 + 1, altura - 1)

            dx = x_original - x1
            dy = y_original - y1

            for canal in range(canais):
                valor_interpolado = (
                    (1 - dx) * (1 - dy) * imagem[y1, x1, canal]
                    + dx * (1 - dy) * imagem[y1, x2, canal]
                    + (1 - dx) * dy * imagem[y2, x1, canal]
                    + dx * dy * imagem[y2, x2, canal]
                )

                nova_imagem[y, x, canal] = valor_interpolado

    return nova_imagem


def rotacao_linear(imagem, angulo):
    imagem = np.array(imagem)
    altura, largura, canais = imagem.shape
    centro_x = largura / 2
    centro_y = altura / 2

    nova_imagem = np.zeros_like(imagem)

    for y in range(altura):
        for x in range(largura):
            x_original = (
                (x - centro_x) * np.cos(angulo)
                - (y - centro_y) * np.sin(angulo)
                + centro_x
            )
            y_original = (
                (x - centro_x) * np.sin(angulo)
                + (y - centro_y) * np.cos(angulo)
                + centro_y
            )

            if 0 <= x_original < largura and 0 <= y_original < altura:
                x1, y1 = int(x_original), int(y_original)
                x2, y2 = min(x1 + 1, largura - 1), min(y1 + 1, altura - 1)

                dx = x_original - x1
                dy = y_original - y1

                for canal in range(canais):
                    valor_interpolado = (
                        (1 - dx) * (1 - dy) * imagem[y1, x1, canal]
                        + dx * (1 - dy) * imagem[y1, x2, canal]
                        + (1 - dx) * dy * imagem[y2, x1, canal]
                        + dx * dy * imagem[y2, x2, canal]
                    )

                    nova_imagem[y, x, canal] = valor_interpolado

    return nova_imagem
