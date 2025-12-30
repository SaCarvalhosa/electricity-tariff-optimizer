# ⚡ Otimizador de Tarifas de Eletricidade Portugal

Uma aplicação web para comparar e otimizar tarifas de eletricidade em Portugal, usando dados oficiais da ERSE e IA local para análise de faturas.

![Python](https://img.shields.io/badge/python-3.10+-blue.svg)
![FastAPI](https://img.shields.io/badge/FastAPI-0.100+-green.svg)
![License](https://img.shields.io/badge/license-EUPL--1.2-blue.svg)

## ✨ Funcionalidades

- **📊 Comparação de Tarifas**: Compara 6000+ tarifas de todos os comercializadores portugueses
- **🔍 Análise de Consumo**: Carregue os seus dados de consumo (CSV) para encontrar a melhor tarifa
- **📄 Análise de Faturas**: Extraia dados automaticamente de faturas PDF ou imagem
- **📧 Geração de Email**: Cria emails prontos a enviar para mudar de fornecedor
- **🤖 IA Local**: Usa Ollama (gratuito) - sem enviar dados para a cloud
- **📅 Normalização**: Converte qualquer período para 30 dias para comparação justa com faturas mensais

## 🚀 Instalação Rápida

### 1. Clone o repositório

```bash
git clone https://github.com/SEU_USER/electricity-tariff-optimizer.git
cd electricity-tariff-optimizer
```

### 2. Instale as dependências

```bash
pip install -r requirements.txt
```

### 3. Instale o Ollama (IA local gratuita)

```bash
# Linux/macOS
curl -fsSL https://ollama.com/install.sh | sh

# Windows - descarregue de https://ollama.com/download
```

### 4. Descarregue um modelo de texto

```bash
# Recomendado (bom equilíbrio qualidade/velocidade)
ollama pull llama3.1:8b

# Melhor qualidade (requer mais VRAM)
ollama pull llama3.1:70b
```

### 5. (Opcional) Suporte a PDFs

```bash
pip install pymupdf
```

### 6. Execute a aplicação

```bash
python app.py
```

Abra http://localhost:8000 no browser.

## 📖 Guia de Utilização

### Carregar Dados de Consumo

A aplicação suporta dois formatos:

**1. Ficheiro Excel e-Redes (recomendado)**
- Exporte os seus consumos de [E-REDES Balcão Digital](https://balcaodigital.e-redes.pt/)
- Formato automático com 15 minutos de intervalo
- O ficheiro é convertido automaticamente para kWh

**2. Ficheiro CSV manual**

O ficheiro CSV deve ter duas colunas:
- `timestamp`: Data/hora (formato ISO ou DD/MM/YYYY HH:MM)
- `kWh`: Consumo em kWh

Exemplo:
```csv
timestamp,kWh
2024-01-01 00:00,0.234
2024-01-01 01:00,0.189
2024-01-01 02:00,0.156
```

### Análise de Faturas

A aplicação suporta três métodos:

1. **PDF** (recomendado): Extração de texto automática
2. **Imagem**: Modelo de visão IA
3. **Manual**: Inserir dados diretamente no formulário

### Entrada Manual de Dados

Se preferir não carregar a fatura, pode inserir os dados manualmente:
- Nome, NIF, CPE
- Fornecedor atual
- Potência contratada
- Valor da última fatura

A aplicação irá:
- Comparar com a melhor tarifa encontrada
- Calcular poupança (ou avisar se a tarifa atual já é melhor!)
- Gerar email pronto a enviar

## 📁 Estrutura do Projeto

```
electricity-tariff-optimizer/
├── app.py                    # 🌐 Aplicação web principal (FastAPI)
├── requirements.txt          # 📦 Dependências Python
├── tariffs.json             # 💰 Base de dados de tarifas ERSE
├── load.csv                 # 📊 Exemplo de dados de consumo
│
├── parse_erse_csv_v2.py     # 🔄 Parser de dados ERSE
├── score_tariffs.py         # 📈 Motor de scoring de tarifas
├── select_contracted_power.py # ⚡ Seleção de potência contratada
│
├── monitor_erse.py          # 👁️ Monitor de atualizações ERSE
├── notify_best_tariff.py    # 📧 Notificações de melhores tarifas
├── scheduler.py             # ⏰ Agendador de tarefas
│
├── Precos_ELEGN.csv         # 📋 Dados de preços ERSE
└── CondComerciais.csv       # 📋 Condições comerciais ERSE
```

## ⚙️ Configuração

### Variáveis de Ambiente

| Variável | Descrição | Default |
|----------|-----------|---------|
| `OLLAMA_HOST` | URL do servidor Ollama | `http://localhost:11434` |
| `OLLAMA_TEXT_MODEL` | Modelo de texto preferido | Auto-detecta (prefere modelos maiores) |
| `OLLAMA_VISION_MODEL` | Modelo de visão | `llava` |
| `ANTHROPIC_API_KEY` | Chave API Anthropic (opcional, pago) | - |

Exemplo:
```bash
OLLAMA_TEXT_MODEL=llama3.1:70b python app.py
```

### Modelos Ollama Recomendados

| Modelo | VRAM | Qualidade | Velocidade |
|--------|------|-----------|------------|
| `llama3.1:8b` | 8GB | ⭐⭐⭐ | ⭐⭐⭐⭐⭐ |
| `llama3.1:70b` | 48GB | ⭐⭐⭐⭐⭐ | ⭐⭐ |
| `mistral:7b` | 6GB | ⭐⭐⭐ | ⭐⭐⭐⭐⭐ |
| `qwen2.5:32b` | 24GB | ⭐⭐⭐⭐ | ⭐⭐⭐ |

A aplicação auto-detecta e usa o melhor modelo disponível.

## 📊 Como Funciona

### Normalização para 30 Dias

Se carregar dados de qualquer período (ex: 91 dias), a aplicação:

1. Calcula o perfil médio de consumo por hora
2. Preserva padrões dia-da-semana vs fim-de-semana (importante para bi-horária/tri-horária)
3. Gera dados sintéticos de 30 dias
4. Permite comparação direta com faturas mensais

### Tipos de Tarifa

| Tipo | Períodos | Melhor para |
|------|----------|-------------|
| **Simples** | Preço único 24h | Consumo constante |
| **Bi-Horária** | Vazio + Fora-de-vazio | Consumo noturno/fim-de-semana |
| **Tri-Horária** | Vazio + Cheias + Ponta | Flexibilidade de horários |

### Comparação Inteligente

A aplicação compara a sua fatura atual com a melhor tarifa encontrada:
- ✅ Se poupar → Recomenda mudança + gera email
- ⚠️ Se a atual for melhor → Avisa e sugere manter

## 🔄 Atualização de Tarifas

As tarifas são obtidas da [ERSE](https://www.erse.pt/). Para atualizar:

```bash
# Descarregar novos dados ERSE
python monitor_erse.py

# Processar e gerar tariffs.json
python parse_erse_csv_v2.py
```

## 🛠️ API Endpoints

| Endpoint | Método | Descrição |
|----------|--------|-----------|
| `/` | GET | Página inicial |
| `/upload` | POST | Carregar CSV de consumo |
| `/results` | GET | Ver ranking de tarifas |
| `/ai` | GET | Página de análise IA |
| `/analyze` | POST | Analisar fatura (PDF/imagem) |
| `/generate-manual` | POST | Gerar email com dados manuais |
| `/recommend` | GET | Recomendação rápida (só texto) |
| `/api/status` | GET | Status dos LLMs disponíveis |

## 🤝 Contribuir

Contribuições são bem-vindas! Por favor:

1. Fork o repositório
2. Crie uma branch (`git checkout -b feature/nova-funcionalidade`)
3. Commit as alterações (`git commit -am 'Adiciona nova funcionalidade'`)
4. Push para a branch (`git push origin feature/nova-funcionalidade`)
5. Abra um Pull Request

## 📝 Licença

Este projeto está licenciado sob a [European Union Public Licence v. 1.2 (EUPL-1.2)](LICENSE).

## ⚠️ Aviso Legal

Esta ferramenta é fornecida apenas para fins informativos. Os preços e condições podem variar. Verifique sempre diretamente com o comercializador antes de efetuar a mudança.

## 🙏 Agradecimentos

- [ERSE](https://www.erse.pt/) - Dados oficiais de tarifas
- [Ollama](https://ollama.com/) - IA local gratuita
- [FastAPI](https://fastapi.tiangolo.com/) - Framework web

---

**Feito com ⚡ em Portugal**
