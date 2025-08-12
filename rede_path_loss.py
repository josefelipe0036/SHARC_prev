import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import seaborn as sns
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from tqdm import tqdm
from sklearn.metrics import mean_absolute_error, mean_squared_error



# --- 1. Preparação dos Dados ---
df = pd.read_csv(r"C:\Users\josef\Documents\UnB\prev_sharc\dados_atualizados_expandido.csv")


np.random.seed(42)


# Criando a coluna de faixas para a estratificação
bins = [-np.inf,150,160,164,168, 172,176, 180, 184, 188, 192, 196, 200, 204, 208, 212, 216, 220 ,np.inf]
df['faixa_path_loss'] = pd.cut(df['path_loss'], bins=bins, right=False)


# --- 2. Amostragem Estratificada 
N = 285000 # Tamanho da amostra desejada 57*quantidade de drop
tamanho_amostra_proporcional = N / len(df) # Converte o tamanho absoluto para uma fração

# Separando X e a coluna de estratificação y_strat
X = df[['bs_x', 'bs_y', 'path_loss']] # Incluindo path_loss para manter os dados juntos
y_strat = df['faixa_path_loss']

# Usando train_test_split para criar a amostra
# O parâmetro 'stratify' garante a amostragem proporcional
df_amostra_proporcional, _ = train_test_split(
    X, # O que queremos amostrar
    train_size=tamanho_amostra_proporcional, 
    stratify=y_strat,
    random_state=42
)

print(f"Amostra estratificada proporcional de tamanho {len(df_amostra_proporcional)} criada.\n")


# --- 3. Verificação ---
print("--- Proporção na Base Original ---")
print(df['faixa_path_loss'].value_counts(normalize=True).sort_index())
print("\n--- Proporção na Amostra Gerada ---")
# Criamos a faixa de novo só para conferir as proporções
df_amostra_proporcional['faixa_path_loss'] = pd.cut(df_amostra_proporcional['path_loss'], bins=bins, right=False)
print(df_amostra_proporcional['faixa_path_loss'].value_counts(normalize=True).sort_index())

# Plotando para confirmação visual
sns.ecdfplot(data=df, x='path_loss', label='CDF Original')
sns.ecdfplot(data=df_amostra_proporcional, x='path_loss', label=f'CDF da Amostra Proporcional (N={N})')
plt.title('Comparação de CDFs (Amostragem Proporcional)')
plt.legend()
plt.grid(True)
plt.show()





######################################
#preparacao dos dados
######################################
# Dados de entrada (Condição) e Saída (Dado Real)
conditions = df_amostra_proporcional[['bs_x', 'bs_y']].values
real_data = df_amostra_proporcional[['path_loss']].values

# Normalização dos dados para o intervalo [-1, 1]
condition_scaler = MinMaxScaler(feature_range=(-1, 1))
data_scaler = MinMaxScaler(feature_range=(-1, 1))

scaled_conditions = condition_scaler.fit_transform(conditions)
scaled_real_data = data_scaler.fit_transform(real_data)





# --- Conversão para Tensores PyTorch ---
# Definir o dispositivo (GPU se disponível, senão CPU)
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Usando dispositivo: {device}")

# Criar tensores
conditions_tensor = torch.FloatTensor(scaled_conditions).to(device)
data_tensor = torch.FloatTensor(scaled_real_data).to(device)

# Criar um DataLoader para gerenciar os lotes
batch_size = 128#64
dataset = TensorDataset(data_tensor, conditions_tensor)
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
print(batch_size)# --- Conversão para Tensores PyTorch ---
# Definir o dispositivo (GPU se disponível, senão CPU)
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Usando dispositivo: {device}")

# Criar tensores
conditions_tensor = torch.FloatTensor(scaled_conditions).to(device)
data_tensor = torch.FloatTensor(scaled_real_data).to(device)

# Criar um DataLoader para gerenciar os lotes
batch_size = 128#64
dataset = TensorDataset(data_tensor, conditions_tensor)
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
print(batch_size)


# --- Parâmetros da cGAN ---
latent_dim = 32      # Dimensão do ruído
conditional_dim = 2  # ue_x, ue_y
data_dim = 1         # path_loss

# --- Construção do Gerador ---
class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(latent_dim + conditional_dim, 128),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(128, 256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(256, data_dim),
            nn.Tanh() # Normalizamos para [-1, 1], então Tanh é a ativação ideal
        )

    def forward(self, noise, conditions):
        # Concatena o ruído e a condição
        merged_input = torch.cat((noise, conditions), -1)
        return self.model(merged_input)

# --- Construção do Discriminador ---
class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(data_dim + conditional_dim, 256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(0.3),
            nn.Linear(256, 128),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(0.3),
            nn.Linear(128, 1),
            nn.Sigmoid() # Classifica como Real (1) ou Falso (0)
        )

    def forward(self, data, conditions):
        # Concatena os dados (reais ou falsos) e a condição
        merged_input = torch.cat((data, conditions), -1)
        return self.model(merged_input)

# Instanciar os modelos e movê-los para o dispositivo
generator = Generator().to(device)
discriminator = Discriminator().to(device)

# Função de perda
adversarial_loss = nn.BCELoss() # Binary Cross Entropy

# Otimizadores
lr = 0.0002#0.0002
optimizer_G = torch.optim.Adam(generator.parameters(), lr=lr, betas=(0.5, 0.999))
optimizer_D = torch.optim.Adam(discriminator.parameters(), lr=lr, betas=(0.5, 0.999))

print("--- Arquitetura do Gerador ---")
print(generator)
print("\n--- Arquitetura do Discriminador ---")
print(discriminator)


# --- Treinamento ---
num_epochs = 80
g_losses = []
d_losses = []

print("\nIniciando o treinamento...")
for epoch in range(num_epochs):
    for i, (real_batch_data, real_batch_conditions) in enumerate(dataloader):
        
        # Labels para os dados reais (1) e falsos (0)
        # .view(-1, 1) garante que o shape é [batch_size, 1]
        valid = torch.ones(real_batch_data.size(0), 1, device=device, requires_grad=False)
        fake = torch.zeros(real_batch_data.size(0), 1, device=device, requires_grad=False)

        # ---------------------
        #  Treinar o Discriminador
        # ---------------------
        optimizer_D.zero_grad() # Limpa os gradientes anteriores
        
        # Gerar um lote de dados falsos
        noise = torch.randn(real_batch_data.size(0), latent_dim, device=device)
        generated_data = generator(noise, real_batch_conditions)
        
        # Calcular a perda para dados reais e falsos
        real_loss = adversarial_loss(discriminator(real_batch_data, real_batch_conditions), valid)
        # Usamos .detach() no dado gerado para não calcular gradientes para o Gerador nesta etapa
        fake_loss = adversarial_loss(discriminator(generated_data.detach(), real_batch_conditions), fake)
        d_loss = (real_loss + fake_loss) / 2
        
        # Backpropagation e atualização dos pesos do Discriminador
        d_loss.backward()
        optimizer_D.step()

        # ---------------------
        #  Treinar o Gerador
        # ---------------------
        optimizer_G.zero_grad()
        
        # Gerar um novo lote de dados e tentar enganar o Discriminador
        noise = torch.randn(real_batch_data.size(0), latent_dim, device=device)
        generated_data = generator(noise, real_batch_conditions)
        
        # O Gerador vence se o Discriminador classificar os dados falsos como reais (label 'valid')
        g_loss = adversarial_loss(discriminator(generated_data, real_batch_conditions), valid)
        
        # Backpropagation e atualização dos pesos do Gerador
        g_loss.backward()
        optimizer_G.step()
    
    # Salvar perdas para plotagem
    d_losses.append(d_loss.item())
    g_losses.append(g_loss.item())

    if epoch % 1 == 0:
        print(
            f"[Época {epoch}/{num_epochs}] [D loss: {d_loss.item():.4f}] [G loss: {g_loss.item():.4f}]"
        )
        
print("Treinamento Concluído!")

# Plotar o histórico de perdas
plt.figure(figsize=(10, 5))
plt.plot(g_losses, label='Generator Loss')
plt.plot(d_losses, label='Discriminator Loss')
plt.title('Histórico de Perdas da cGAN (PyTorch)')
plt.xlabel('Época')
plt.ylabel('Perda (Loss)')
plt.legend()
plt.grid(True)
plt.show()





###################################################################
#avaliando a rede
#a funcao gerara path loss pela GAN
def gerar_pathloss_pytorch(ue_x, ue_y):
    """
    Função para gerar um valor de path_loss com o modelo PyTorch treinado.
    """
    # Coloca o gerador em modo de avaliação
    generator.eval()
    
    # Desativa o cálculo de gradientes para inferência
    with torch.no_grad():
        # 1. Preparar os dados de entrada (condição)
        coords = np.array([[ue_x, ue_y]])
        
        # 2. Escalar os dados
        scaled_coords = condition_scaler.transform(coords)
        
        # 3. Converter para tensor e mover para o dispositivo
        coords_tensor = torch.FloatTensor(scaled_coords).to(device)
        
        # 4. Preparar o ruído
        noise = torch.randn(1, latent_dim, device=device)
        
        # 5. Gerar o path_loss escalado
        scaled_generated_pl = generator(noise, coords_tensor)
        
        # 6. Mover resultado de volta para a CPU e converter para numpy
        generated_pl_numpy = scaled_generated_pl.cpu().numpy()
        
        # 7. Reverter a escala
        original_pl = data_scaler.inverse_transform(generated_pl_numpy)
        
        return original_pl[0][0]

# --- Exemplo de Uso ---
exemplo_x, exemplo_y = 500, 500
path_loss_gerado = gerar_pathloss_pytorch(exemplo_x, exemplo_y)
print(f"\nPara as coordenadas (x={exemplo_x}, y={exemplo_y}), o path_loss gerado foi: {path_loss_gerado:.2f} dB")

# --- Verificação Visual (igual ao código anterior) ---
generator.eval()
with torch.no_grad():
    noise = torch.randn(1000, latent_dim, device=device)
    # Pega 1000 condições aleatórias dos dados de treino
    random_indices = np.random.randint(0, len(conditions_tensor), 1000)
    sample_conditions = conditions_tensor[random_indices]
    
    generated_samples_scaled = generator(noise, sample_conditions)
    generated_samples = data_scaler.inverse_transform(generated_samples_scaled.cpu().numpy())

# Comparar a distribuição dos dados reais e gerados
plt.figure(figsize=(12, 6))
sns.kdeplot(df_amostra_proporcional['path_loss'], label='Path Loss Real (da Amostra)', color='blue', fill=True)
sns.kdeplot(generated_samples.flatten(), label='Path Loss Gerado (pela cGAN)', color='red', fill=True)
plt.title('Comparação da Distribuição de Path Loss Real vs. Gerado (PyTorch)')
plt.xlabel('Path Loss (dB)')
plt.ylabel('Densidade')
plt.legend()
plt.grid(True)
plt.show()





#testando a Gan com os dados que foram treinados

# --- Passo 1: Gerar uma Amostra Grande de Dados ---

# Define o número de amostras a serem geradas. Um bom número é o tamanho da base de treino.
num_generated_samples = len(df_amostra_proporcional)
print(f"Gerando {num_generated_samples} amostras de path_loss com a cGAN...")

# Coloca o gerador em modo de avaliação
generator.eval()

# Desativa o cálculo de gradientes para acelerar a inferência
with torch.no_grad():
    # Pega condições aleatórias dos dados de treinamento para gerar path_loss correspondente
    random_indices = np.random.randint(0, len(conditions_tensor), num_generated_samples)
    sample_conditions = conditions_tensor[random_indices]
    
    # Gera ruído aleatório
    noise = torch.randn(num_generated_samples, latent_dim, device=device)
    
    # Gera os dados escalados
    generated_samples_scaled = generator(noise, sample_conditions)
    
    # Move os dados para a CPU (se estiver na GPU) e converte para numpy
    generated_samples_numpy = generated_samples_scaled.cpu().numpy()
    
    # Inverte a escala para obter os valores originais de path_loss
    path_loss_gerado = data_scaler.inverse_transform(generated_samples_numpy).flatten()

print("Amostras geradas com sucesso!")


# --- Passo 2: Preparar os Dados para Plotagem ---

# Dados reais da sua amostra original
path_loss_real = df_amostra_proporcional['path_loss'].values


# --- Passo 3: Plotar as CDFs para Comparação ---

plt.style.use('seaborn-v0_8-whitegrid') # Define um estilo bonito para o gráfico
plt.figure(figsize=(12, 7))

# Plotar a CDF dos dados reais
sns.ecdfplot(x=path_loss_real, linewidth=2.5, 
             label='CDF dos Dados Reais')

# Plotar a CDF dos dados gerados pela GAN
sns.ecdfplot(x=path_loss_gerado, linewidth=2.5, linestyle='--',
             label='CDF dos Dados Gerados pela cGAN')

plt.title('Comparação de CDFs: Dados Reais vs. Dados Gerados', fontsize=16)
plt.xlabel('Path Loss (dB)', fontsize=12)
plt.ylabel('Probabilidade Cumulativa (P[Path Loss ≤ x])', fontsize=12)
plt.legend(fontsize=11)
plt.show()









#funcao para gerar path loss em lotes para nao estourar a memoria ram


def gerar_em_batches(df, batch_size=512):
    """
    Gera path loss com a cGAN treinada, processando em batches para evitar estouro de memória.
    """
    df_resultado = df.copy()
    coords = df[['ue_x', 'ue_y']].values
    total = len(coords)
    path_loss_gerado_list = []

    generator.eval()
    with torch.no_grad():
        for i in tqdm(range(0, total, batch_size)):
            # Seleciona o batch
            batch_coords = coords[i:i+batch_size]

            # Escala
            scaled_coords = condition_scaler.transform(batch_coords)
            coords_tensor = torch.FloatTensor(scaled_coords).to(device)

            # Ruído
            noise = torch.randn(len(batch_coords), latent_dim, device=device)

            # Geração
            scaled_generated = generator(noise, coords_tensor)

            # Volta para CPU e desscale
            pl_numpy = scaled_generated.cpu().numpy()
            pl_original = data_scaler.inverse_transform(pl_numpy)

            # Armazena
            path_loss_gerado_list.extend(pl_original.flatten())

    df_resultado['path_loss_gerado'] = path_loss_gerado_list
    return df_resultado


##############
df_com_gerados = gerar_em_batches(df, batch_size=100000)



# === Métricas ===
y_true = df_com_gerados['path_loss'].values
y_pred = df_com_gerados['path_loss_gerado'].values

mae = mean_absolute_error(y_true, y_pred)
mse = mean_squared_error(y_true, y_pred)
rmse = np.sqrt(mse)


print(f"   MAE  = {mae:.4f} dB")
print(f"   MSE  = {mse:.4f} dB²")
print(f"   RMSE = {rmse:.4f} dB")


# Ordenar valores
y_true_sorted = np.sort(y_true)
y_pred_sorted = np.sort(y_pred)

# Construir a CDF manualmente
cdf_true = np.arange(1, len(y_true_sorted)+1) / len(y_true_sorted)
cdf_pred = np.arange(1, len(y_pred_sorted)+1) / len(y_pred_sorted)

plt.figure(figsize=(10, 6))
plt.plot(y_true_sorted, cdf_true, label='Path Loss Real', color='blue', linewidth=2)
plt.plot(y_pred_sorted, cdf_pred, label='Path Loss Gerado (cGAN)', color='red', linewidth=2)

plt.title('Função de Distribuição Acumulada (CDF)\nPath Loss Real vs. Gerado')
plt.xlabel('Path Loss (dB)')
plt.ylabel('Probabilidade Acumulada')
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()
