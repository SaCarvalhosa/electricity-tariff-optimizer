# app.py - Portuguese Electricity Tariff Optimizer Web Application
# Enhanced with LLM-powered receipt analysis (Ollama FREE or Anthropic paid)
"""
Run with: python app.py
Open: http://localhost:8000

For AI features (FREE):
  1. Install Ollama: https://ollama.ai
  2. Pull a vision model: ollama pull llava
  3. Start Ollama (usually auto-starts)
  4. Run this app

For AI features (PAID alternative):
  export ANTHROPIC_API_KEY=your-key-here
"""

import io, os, json, base64, re
from pathlib import Path
from typing import Optional
import numpy as np
import pandas as pd
from fastapi import FastAPI, UploadFile, File, Request
from fastapi.responses import HTMLResponse, JSONResponse
import uvicorn

# ============================================================
# CONFIGURATION
# ============================================================

TARIFFS_PATH = Path("tariffs.json")
DEFAULT_LOAD_PATH = Path("load.csv")
OFFERS_CSV_PATH = Path("CondComerciais.csv")
STANDARD_KVA = [1.15, 2.30, 3.45, 4.60, 5.75, 6.90, 10.35, 13.80, 17.25, 20.70]

# LLM Config - Ollama (free) is primary, Anthropic (paid) is fallback
OLLAMA_HOST = os.environ.get("OLLAMA_HOST", "http://localhost:11434")
OLLAMA_VISION_MODEL = os.environ.get("OLLAMA_VISION_MODEL", "llava")
OLLAMA_TEXT_MODEL = os.environ.get("OLLAMA_TEXT_MODEL", "llama3.1:8b-instruct-q8_0")
ANTHROPIC_API_KEY = os.environ.get("ANTHROPIC_API_KEY", "")

# ============================================================
# LLM CLIENT - Ollama (free) first, Anthropic (paid) fallback
# ============================================================

def check_ollama() -> dict:
    """Check Ollama availability and models."""
    try:
        import requests
        r = requests.get(f"{OLLAMA_HOST}/api/tags", timeout=3)
        if r.status_code == 200:
            models = [m["name"] for m in r.json().get("models", [])]
            # Find vision model
            vision = ""
            for vm in ["llava", "llava:latest", "llava:13b", "llama3.2-vision", "bakllava", "moondream"]:
                for m in models:
                    if vm.split(":")[0] in m:
                        vision = m
                        break
                if vision:
                    break
            # Find text model - prefer larger/better models first
            text = ""
            # Priority order: env var, then larger models first
            text_priority = [
                OLLAMA_TEXT_MODEL,  # User preference first
                # Large models (best quality)
                "llama3.1:70b", "qwen2.5:72b", "llama3:70b", "mixtral:8x22b",
                # Medium models (good balance)
                "llama3.1:8b", "qwen2.5:32b", "mixtral:8x7b", "llama3.1",
                # Small models (faster)
                "llama3", "mistral", "gemma", "phi3"
            ]
            for tm in text_priority:
                for m in models:
                    # Check if model name matches (handle tags like :latest, :70b-instruct-q4_0)
                    tm_base = tm.split(":")[0]
                    m_base = m.split(":")[0]
                    if tm_base == m_base or tm == m:
                        text = m
                        break
                if text:
                    break
            return {"available": True, "models": models, "vision": vision, "text": text or (models[0] if models else "")}
    except:
        pass
    return {"available": False, "models": [], "vision": "", "text": ""}


def get_llm_status() -> dict:
    """Get LLM availability status."""
    ollama = check_ollama()
    return {
        "ollama_available": ollama["available"],
        "ollama_models": ollama["models"],
        "ollama_vision": ollama["vision"],
        "ollama_text": ollama["text"],
        "anthropic_available": bool(ANTHROPIC_API_KEY),
        "any_available": ollama["available"] or bool(ANTHROPIC_API_KEY),
        "vision_available": bool(ollama["vision"]) or bool(ANTHROPIC_API_KEY),
    }


def parse_llm_response(text: str, best_provider: str, best_code: str, best_total: float = 0) -> dict:
    """Parse natural language LLM response into structured data."""
    
    # Helper to find value after a label
    def find_value(patterns, text):
        for pattern in patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                return match.group(1).strip()
        return ""
    
    # List of known Portuguese electricity providers
    known_providers = ["EDP", "Iberdrola", "Endesa", "Galp", "Goldenergy", "Luzboa", 
                       "MEO Energia", "MEO", "Muon", "SU Eletricidade", "Repsol", 
                       "Plenitude", "Coopernico", "Audax", "Enat", "YU Energy"]
    
    # Try to find provider from known list first
    provider = ""
    text_upper = text.upper()
    for p in known_providers:
        if p.upper() in text_upper:
            provider = p
            break
    
    # If not found, try regex patterns
    if not provider:
        provider = find_value([
            r'fornecedor[:\s]+([A-Za-zÀ-ÿ\s]+?)(?:\n|,|\.)',
        ], text)
    
    customer = find_value([
        r'cliente[:\s]+([A-Za-zÀ-ÿ\s]+?)(?:\n|,|NIF)',
        r'nome[:\s]+([A-Za-zÀ-ÿ\s]+?)(?:\n|,|\.)',
    ], text)
    
    nif = find_value([
        r'NIF[:\s]*(\d{9})',
        r'(\d{9})',
    ], text)
    
    cpe = find_value([
        r'CPE[:\s]*(PT\d+)',
        r'(PT0002\d+)',
    ], text)
    
    power = find_value([
        r'potência[:\s]*([\d,\.]+\s*kVA)',
        r'([\d,\.]+)\s*kVA',
    ], text)
    
    total = find_value([
        r'total[:\s]*([\d,\.]+)\s*€',
        r'valor[:\s]*([\d,\.]+)\s*€',
        r'fatura[:\s]*([\d,\.]+)\s*€',
    ], text)
    
    # Find savings mentions
    annual_savings = find_value([
        r'poupança anual[:\s]*([\d,\.]+\s*€)',
        r'anual[:\s]*([\d,\.]+\s*€)',
        r'por ano[:\s]*([\d,\.]+\s*€)',
        r'(\d+[\d,\.]*\s*€)\s*(?:por ano|anual)',
    ], text)
    
    # Extract recommendation (look for section C or paragraphs about savings)
    rec_match = re.search(r'(?:C\)|Recomendação)[:\s]*([\s\S]+?)(?=D\)|Email|Assunto|Exmo|Prezad|$)', text, re.IGNORECASE)
    if rec_match:
        recommendation = rec_match.group(1).strip()
    else:
        # Take paragraphs that discuss savings/switching
        paragraphs = [p.strip() for p in text.split('\n') if len(p.strip()) > 30 and ('€' in p or 'poup' in p.lower() or 'mud' in p.lower() or 'recomend' in p.lower())]
        recommendation = '\n\n'.join(paragraphs[:4]) if paragraphs else ""
    
    # Clean up recommendation - remove any repeated patterns
    if recommendation:
        lines = recommendation.split('\n')
        seen = set()
        clean_lines = []
        for line in lines:
            line_key = line.strip()[:50]  # Use first 50 chars as key
            if line_key and line_key not in seen:
                seen.add(line_key)
                clean_lines.append(line)
        recommendation = '\n'.join(clean_lines[:10])  # Max 10 lines
    
    # Compare extracted bill total with best tariff to check if switching makes sense
    is_cheaper = None
    comparison_note = ""
    if total and best_total > 0:
        try:
            current_val = float(total.replace(',', '.').replace('€', '').strip())
            if current_val > 0:
                diff = current_val - best_total
                if diff > 0:
                    is_cheaper = True
                    comparison_note = f"POUPANÇA: {diff:.2f}€/mês ({diff*12:.2f}€/ano)"
                else:
                    is_cheaper = False
                    comparison_note = f"ATENÇÃO: A sua fatura atual ({current_val:.2f}€) já é mais barata que a melhor tarifa ({best_total:.2f}€)"
        except:
            pass
    
    # Generate email template based on comparison
    if is_cheaper == False:
        # Current tariff is better - warn user
        email_draft = f"""⚠️ NOTA: A sua fatura atual ({total}€) parece ser mais competitiva que esta oferta ({best_total:.2f}€/mês).
Verifique se o consumo corresponde ao período analisado.

Se mesmo assim quiser pedir informações:

---

Assunto: Pedido de Informação - Oferta {best_code}

Exmo(a) Senhor(a),

Venho por este meio solicitar informações sobre a oferta {best_code} da {best_provider}.

Atualmente sou cliente da {provider or '[fornecedor atual]'} e gostaria de comparar as condições.

Os meus dados são:

Nome: {customer or '[o seu nome]'}
NIF: {nif or '[o seu NIF]'}
CPE: {cpe or '[o seu código CPE]'}
Potência contratada: {power or '[potência atual]'}

Agradeço o envio de informação detalhada.

Com os melhores cumprimentos,
{customer or '[o seu nome]'}"""
    else:
        email_draft = f"""Assunto: Pedido de Adesão - Oferta {best_code}

Exmo(a) Senhor(a),

Venho por este meio manifestar o meu interesse em aderir à oferta {best_code} da {best_provider}.

Atualmente sou cliente da {provider or '[fornecedor atual]'} e pretendo efetuar a mudança de fornecedor de eletricidade para a vossa empresa.

Os meus dados para o processo de mudança são:

Nome: {customer or '[o seu nome]'}
NIF: {nif or '[o seu NIF]'}
CPE: {cpe or '[o seu código CPE - consulte a sua fatura atual]'}
Potência contratada: {power or '[potência atual]'}
Morada de fornecimento: [a sua morada]

Agradeço que me contactem para finalizar o processo de adesão e esclarecer quaisquer dúvidas sobre a mudança.

Com os melhores cumprimentos,
{customer or '[o seu nome]'}

Contacto: [o seu email/telefone]"""
    
    # Update recommendation based on comparison
    if is_cheaper == False and recommendation:
        recommendation = f"⚠️ {comparison_note}\n\n{recommendation}"
    elif is_cheaper == True and comparison_note:
        recommendation = f"✅ {comparison_note}\n\n{recommendation}" if recommendation else comparison_note

    return {
        "success": True,
        "extracted_info": {
            "current_provider": provider or "Não identificado",
            "customer_name": customer or "Não identificado",
            "customer_nif": nif or "Não identificado",
            "cpe": cpe or "Não identificado",
            "contracted_power": power or "Não identificado",
            "total_amount_eur": total or "Não identificado",
        },
        "comparison": {
            "annual_savings": comparison_note if comparison_note else (annual_savings or "Ver recomendação"),
        },
        "recommendation": recommendation or "Com base na análise do seu consumo, a tarifa recomendada oferece um preço mais competitivo. Recomendamos que considere a mudança para reduzir os seus custos de eletricidade.",
        "email_draft": email_draft,
    }


def extract_text_from_pdf(pdf_bytes: bytes) -> str:
    """Extract text from PDF using pymupdf (fitz)."""
    try:
        import fitz  # pymupdf
        doc = fitz.open(stream=pdf_bytes, filetype="pdf")
        text = ""
        for page in doc:
            text += page.get_text()
        doc.close()
        return text.strip()
    except ImportError:
        return ""
    except Exception as e:
        return ""


def extract_text_from_image(image_bytes: bytes) -> str:
    """Extract text from image using OCR (tesseract via pytesseract)."""
    try:
        import pytesseract
        from PIL import Image
        import io
        img = Image.open(io.BytesIO(image_bytes))
        text = pytesseract.image_to_string(img, lang='por')  # Portuguese
        return text.strip()
    except ImportError:
        return ""
    except Exception as e:
        return ""


def analyze_bill_text_with_llm(text: str, best_tariff: dict, stats: dict) -> dict:
    """Analyze extracted bill text with a TEXT LLM (more reliable than vision)."""
    status = get_llm_status()
    
    if not text:
        return {"success": False, "error": "Não foi possível extrair texto do documento."}
    
    best_provider = best_tariff.get('provider', 'N/A')
    best_code = best_tariff.get('cod_proposta', 'N/A')
    best_total = best_tariff.get('total_eur', 0)
    best_avg = best_tariff.get('avg_price', 0)
    days = stats.get('days', 0)
    total_kwh = stats.get('total_kwh', 0)
    
    prompt = f"""Analisa este texto extraído de uma fatura de eletricidade portuguesa e extrai a seguinte informação em formato estruturado:

TEXTO DA FATURA:
{text[:4000]}

EXTRAI (responde APENAS com estes campos, um por linha):
FORNECEDOR: [nome do fornecedor - ex: EDP, Iberdrola, Endesa]
CLIENTE: [nome do cliente]
NIF: [número de 9 dígitos]
CPE: [código que começa com PT0002]
POTENCIA: [potência em kVA]
TOTAL: [valor total da fatura em euros]
PERIODO: [período de faturação]

Se não encontrares algum campo, escreve "N/A"."""

    response_text = ""
    llm_used = ""
    
    # Try Ollama text model (much better for this than vision)
    if status["ollama_text"]:
        try:
            import ollama
            r = ollama.chat(
                model=status["ollama_text"], 
                messages=[{"role": "user", "content": prompt}],
                options={"temperature": 0.1, "num_predict": 500}
            )
            response_text = r["message"]["content"]
            llm_used = f"Ollama ({status['ollama_text']})"
        except Exception as e:
            if not ANTHROPIC_API_KEY:
                return {"success": False, "error": f"Erro Ollama: {e}"}
    
    # Fallback to Anthropic
    if not response_text and ANTHROPIC_API_KEY:
        try:
            import anthropic
            r = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY).messages.create(
                model="claude-sonnet-4-20250514", 
                max_tokens=500, 
                messages=[{"role": "user", "content": prompt}]
            )
            response_text = r.content[0].text
            llm_used = "Anthropic"
        except Exception as e:
            return {"success": False, "error": f"Erro Anthropic: {e}"}
    
    if not response_text:
        return {"success": False, "error": "Nenhum LLM disponível"}
    
    # Parse the structured response
    def extract_field(text, field):
        match = re.search(rf'{field}:\s*(.+?)(?:\n|$)', text, re.IGNORECASE)
        if match:
            val = match.group(1).strip()
            return val if val.upper() != "N/A" else ""
        return ""
    
    provider = extract_field(response_text, "FORNECEDOR")
    customer = extract_field(response_text, "CLIENTE")
    nif = extract_field(response_text, "NIF")
    cpe = extract_field(response_text, "CPE")
    power = extract_field(response_text, "POTENCIA")
    total = extract_field(response_text, "TOTAL")
    
    # Use parse_llm_response to generate email and comparison
    # But we'll build our own result with the extracted data
    result = parse_llm_response(response_text, best_provider, best_code, best_total)
    
    # Override with our cleaner extraction
    result["extracted_info"] = {
        "current_provider": provider or "Não identificado",
        "customer_name": customer or "Não identificado", 
        "customer_nif": nif or "Não identificado",
        "cpe": cpe or "Não identificado",
        "contracted_power": power or "Não identificado",
        "total_amount_eur": total or "Não identificado",
    }
    result["llm"] = llm_used + " (texto)"
    
    return result


def analyze_receipt_with_llm(image_data: bytes, media_type: str, best_tariff: dict, stats: dict) -> dict:
    """Analyze receipt - tries text extraction first, then vision models."""
    status = get_llm_status()
    
    # STRATEGY 1: Try text extraction first (more reliable for documents)
    extracted_text = ""
    
    if media_type == "application/pdf":
        extracted_text = extract_text_from_pdf(image_data)
        if extracted_text and len(extracted_text) > 100:
            # Good text extraction from PDF - use text LLM
            result = analyze_bill_text_with_llm(extracted_text, best_tariff, stats)
            if result.get("success"):
                result["method"] = "PDF texto"
                return result
    
    # Try OCR on images
    if media_type in ["image/jpeg", "image/png", "image/jpg", "image/webp"]:
        extracted_text = extract_text_from_image(image_data)
        if extracted_text and len(extracted_text) > 100:
            # Good OCR extraction - use text LLM
            result = analyze_bill_text_with_llm(extracted_text, best_tariff, stats)
            if result.get("success"):
                result["method"] = "OCR texto"
                return result
    
    # STRATEGY 2: Fall back to vision model
    if not status["vision_available"]:
        # No vision model and text extraction failed
        if media_type == "application/pdf":
            return {"success": False, "error": "Instale pymupdf para PDFs: pip install pymupdf"}
        return {"success": False, "error": "Instale pytesseract para OCR ou um modelo de visão: ollama pull llava"}
    
    # Vision models don't support PDF
    if media_type == "application/pdf":
        return {"success": False, "error": "PDFs requerem pymupdf. Instale: pip install pymupdf"}
    
    if media_type not in ["image/jpeg", "image/png", "image/jpg", "image/webp"]:
        return {"success": False, "error": f"Formato não suportado: {media_type}. Use JPG, PNG ou PDF."}
    
    best_provider = best_tariff.get('provider', 'N/A')
    best_code = best_tariff.get('cod_proposta', 'N/A')
    best_total = best_tariff.get('total_eur', 0)
    best_avg = best_tariff.get('avg_price', 0)
    days = stats.get('days', 0)
    total_kwh = stats.get('total_kwh', 0)
    
    # Simpler prompt for local vision models
    prompt = f"""Olha para esta fatura de eletricidade portuguesa e extrai a seguinte informação:

1. Nome do fornecedor atual (ex: EDP, Iberdrola, Endesa)
2. Nome do cliente
3. NIF do cliente  
4. Código CPE (começa com PT0002)
5. Potência contratada (em kVA)
6. Valor total da fatura (em euros)
7. Período de faturação

Depois compara com esta oferta melhor que encontrei:
- Novo fornecedor: {best_provider}
- Código oferta: {best_code}
- Custo estimado: {best_total:.2f}€ para {days} dias ({total_kwh:.0f} kWh)
- Preço médio: {best_avg:.4f}€/kWh

Por favor responde com:
A) Os dados extraídos da fatura
B) A poupança estimada (anual)
C) Uma recomendação de 2 parágrafos
D) Um email formal em português para enviar ao {best_provider} a pedir para mudar, incluindo os dados do cliente"""

    # Try Ollama first (FREE)
    if status["ollama_vision"]:
        # Ollama vision models don't support PDF
        if media_type == "application/pdf":
            return {"success": False, "error": "PDFs não suportados. Por favor carregue uma imagem (JPG ou PNG) da fatura."}
        
        # Check for unsupported formats
        if media_type not in ["image/jpeg", "image/png", "image/jpg", "image/webp"]:
            return {"success": False, "error": f"Formato não suportado: {media_type}. Use JPG ou PNG."}
        
        try:
            import ollama
            import tempfile
            
            # Determine file extension from media type
            ext = ".png" if media_type == "image/png" else ".webp" if media_type == "image/webp" else ".jpg"
            
            with tempfile.NamedTemporaryFile(suffix=ext, delete=False) as f:
                f.write(image_data)
                temp_path = f.name
            
            try:
                response = ollama.chat(
                    model=status["ollama_vision"],
                    messages=[
                        {"role": "user", "content": prompt, "images": [temp_path]}
                    ],
                    options={"temperature": 0.2}
                )
            finally:
                os.unlink(temp_path)  # Clean up
            
            text = response["message"]["content"]
            
            # Parse the natural language response into structured data
            result = parse_llm_response(text, best_provider, best_code, best_total)
            result["llm"] = f"Ollama ({status['ollama_vision']})"
            return result
            
        except Exception as e:
            error_msg = str(e)
            if not ANTHROPIC_API_KEY:
                return {"success": False, "error": f"Ollama erro: {error_msg}"}
    
    # Fallback to Anthropic (PAID)
    if ANTHROPIC_API_KEY:
        try:
            import anthropic
            client = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)
            response = client.messages.create(
                model="claude-sonnet-4-20250514",
                max_tokens=4096,
                messages=[{"role": "user", "content": [
                    {"type": "image", "source": {"type": "base64", "media_type": media_type, "data": base64.b64encode(image_data).decode()}},
                    {"type": "text", "text": prompt}
                ]}]
            )
            text = response.content[0].text
            result = parse_llm_response(text, best_provider, best_code, best_total)
            result["llm"] = "Anthropic (Claude)"
            return result
        except Exception as e:
            return {"success": False, "error": f"Anthropic erro: {e}"}
    
    return {"error": "Nenhum LLM disponível"}


def generate_recommendation(best_tariff: dict, stats: dict) -> dict:
    """Generate text recommendation (no image)."""
    status = get_llm_status()
    if not status["any_available"]:
        return {"error": "Nenhum LLM disponível. Inicie Ollama ou configure ANTHROPIC_API_KEY."}
    
    prompt = f"""Recomendação para tarifa de eletricidade em português:

Tarifa: {best_tariff.get('provider')} - {best_tariff.get('cod_proposta')}
Tipo: {best_tariff.get('kind_label')} | Potência: {best_tariff.get('power_kva')} kVA
Custo: {best_tariff.get('total_eur'):.2f}€ ({stats.get('days')} dias) | Médio: {best_tariff.get('avg_price'):.4f}€/kWh
Consumo: {stats.get('total_kwh'):.0f} kWh

Escreve 2 parágrafos explicando porque esta tarifa é boa e diferenças entre simples/bi-horária/tri-horária."""

    # Try Ollama (FREE)
    if status["ollama_text"]:
        try:
            import ollama
            r = ollama.chat(model=status["ollama_text"], messages=[{"role": "user", "content": prompt}])
            return {"success": True, "recommendation": r["message"]["content"], "llm": f"Ollama ({status['ollama_text']})"}
        except Exception as e:
            if not ANTHROPIC_API_KEY:
                return {"success": False, "error": str(e)}
    
    # Fallback Anthropic (PAID)
    if ANTHROPIC_API_KEY:
        try:
            import anthropic
            r = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY).messages.create(
                model="claude-sonnet-4-20250514", max_tokens=1024, messages=[{"role": "user", "content": prompt}])
            return {"success": True, "recommendation": r.content[0].text, "llm": "Anthropic"}
        except Exception as e:
            return {"success": False, "error": str(e)}
    
    return {"error": "Sem LLM"}


def generate_email_from_manual_data(user_data: dict, best_tariff: dict, stats: dict) -> dict:
    """Generate email and recommendation from manually entered data."""
    status = get_llm_status()
    
    # Extract user data
    customer = user_data.get("customer_name", "").strip()
    nif = user_data.get("nif", "").strip()
    cpe = user_data.get("cpe", "").strip()
    current_provider = user_data.get("current_provider", "").strip()
    power = user_data.get("power", "").strip()
    current_bill = user_data.get("current_bill", "").strip()
    email_contact = user_data.get("email", "").strip()
    phone = user_data.get("phone", "").strip()
    address = user_data.get("address", "").strip()
    
    best_provider = best_tariff.get('provider', '')
    best_code = best_tariff.get('cod_proposta', '')
    best_total = best_tariff.get('total_eur', 0)
    best_avg = best_tariff.get('avg_price', 0)
    days = stats.get('days', 0)
    total_kwh = stats.get('total_kwh', 0)
    
    # Calculate savings if current bill provided
    savings_text = ""
    annual_savings = ""
    is_cheaper = None  # None = unknown, True = saves money, False = more expensive
    monthly_diff = 0
    
    if current_bill:
        try:
            current_val = float(current_bill.replace(',', '.').replace('€', '').strip())
            monthly_current = current_val
            monthly_new = best_total  # Already normalized to 30 days
            monthly_diff = monthly_current - monthly_new
            annual_diff = monthly_diff * 12
            
            if monthly_diff > 0:
                # New tariff is CHEAPER
                is_cheaper = True
                annual_savings = f"{annual_diff:.2f}€"
                savings_text = f"POUPANÇA: Com base na sua fatura atual de {current_val:.2f}€ e a melhor tarifa encontrada ({monthly_new:.2f}€), pode poupar aproximadamente {monthly_diff:.2f}€ por mês, ou seja, cerca de {annual_diff:.2f}€ por ano."
            else:
                # New tariff is MORE EXPENSIVE or same
                is_cheaper = False
                extra_cost = abs(monthly_diff)
                annual_extra = abs(annual_diff)
                savings_text = f"ATENÇÃO: A sua fatura atual ({current_val:.2f}€) já é mais barata que a melhor tarifa encontrada ({monthly_new:.2f}€). Diferença: +{extra_cost:.2f}€/mês (+{annual_extra:.2f}€/ano). A sua tarifa atual parece ser competitiva - verifique se o consumo da fatura corresponde ao período analisado."
                annual_savings = f"Sem poupança (atual é mais barato)"
        except:
            pass
    
    # Generate email template (only if switching makes sense or user wants it anyway)
    if is_cheaper == False:
        # Current tariff is better - provide informational email but with warning
        email_draft = f"""⚠️ NOTA: A sua tarifa atual ({current_bill}€/mês) parece ser mais competitiva que esta oferta ({best_total:.2f}€/mês).

Se mesmo assim quiser pedir informações, pode usar o email abaixo:

---

Assunto: Pedido de Informação - Oferta {best_code}

Exmo(a) Senhor(a),

Venho por este meio solicitar informações sobre a oferta {best_code} da {best_provider}.

Atualmente sou cliente da {current_provider or '[fornecedor atual]'} e gostaria de comparar as condições.

Os meus dados são:

Nome: {customer}
NIF: {nif}
CPE: {cpe}
Potência contratada: {power} kVA
Morada de fornecimento: {address or '[a sua morada]'}

Agradeço o envio de informação detalhada sobre preços e condições.

Com os melhores cumprimentos,
{customer}

Email: {email_contact or '[o seu email]'}
Telefone: {phone or '[o seu telefone]'}"""
    else:
        email_draft = f"""Assunto: Pedido de Adesão - Oferta {best_code}

Exmo(a) Senhor(a),

Venho por este meio manifestar o meu interesse em aderir à oferta {best_code} da {best_provider}.

Atualmente sou cliente da {current_provider or '[fornecedor atual]'} e pretendo efetuar a mudança de fornecedor de eletricidade para a vossa empresa.

Os meus dados para o processo de mudança são:

Nome: {customer}
NIF: {nif}
CPE: {cpe}
Potência contratada: {power} kVA
Morada de fornecimento: {address or '[a sua morada]'}

Agradeço que me contactem para finalizar o processo de adesão e esclarecer quaisquer dúvidas sobre a mudança.

Com os melhores cumprimentos,
{customer}

Email: {email_contact or '[o seu email]'}
Telefone: {phone or '[o seu telefone]'}"""

    # Try to get LLM recommendation
    recommendation = ""
    llm_used = "Template"
    
    if status["any_available"]:
        # Build an explicit prompt based on whether there are savings or not
        if is_cheaper == True:
            prompt = f"""Escreve uma recomendação positiva em português (2-3 parágrafos curtos) para um cliente que VAI POUPAR dinheiro ao mudar de fornecedor.

Dados:
- Cliente: {customer}
- Fornecedor atual: {current_provider} - paga {current_bill}€/mês
- Novo fornecedor: {best_provider} (oferta {best_code}) - pagará {best_total:.2f}€/mês
- POUPANÇA: {abs(monthly_diff):.2f}€/mês = {abs(monthly_diff)*12:.2f}€/ano

Sê entusiasmante mas profissional. Explica que a mudança é gratuita e sem cortes no fornecimento."""

        elif is_cheaper == False:
            prompt = f"""Escreve uma análise honesta em português (2-3 parágrafos curtos) para um cliente cuja tarifa atual JÁ É MAIS BARATA que as alternativas encontradas.

Dados:
- Cliente: {customer}
- Fornecedor atual: {current_provider} - paga {current_bill}€/mês
- Melhor alternativa: {best_provider} (oferta {best_code}) - custaria {best_total:.2f}€/mês
- DIFERENÇA: A tarifa atual é {abs(monthly_diff):.2f}€/mês MAIS BARATA

IMPORTANTE: NÃO recomende a mudança! Diz ao cliente que a tarifa atual parece ser competitiva. 
Sugere apenas verificar se o consumo na fatura corresponde ao período analisado ({total_kwh:.0f} kWh em 30 dias).
Se o consumo variar sazonalmente, os resultados podem ser diferentes."""

        else:
            prompt = f"""Escreve uma recomendação em português (2-3 parágrafos curtos) sobre a melhor tarifa encontrada.

Dados:
- Cliente: {customer}
- Fornecedor atual: {current_provider}
- Melhor tarifa: {best_provider} (oferta {best_code})
- Custo estimado: {best_total:.2f}€ para 30 dias ({total_kwh:.0f} kWh)
- Preço médio: {best_avg:.4f}€/kWh

Explica os benefícios e que a mudança é gratuita e sem cortes."""

        if status["ollama_text"]:
            try:
                import ollama
                r = ollama.chat(model=status["ollama_text"], messages=[{"role": "user", "content": prompt}], options={"temperature": 0.3, "num_predict": 400})
                recommendation = r["message"]["content"]
                llm_used = f"Ollama ({status['ollama_text']})"
            except:
                pass
        
        if not recommendation and ANTHROPIC_API_KEY:
            try:
                import anthropic
                r = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY).messages.create(
                    model="claude-sonnet-4-20250514", max_tokens=400, messages=[{"role": "user", "content": prompt}])
                recommendation = r.content[0].text
                llm_used = "Anthropic"
            except:
                pass
    
    # Fallback recommendation based on comparison
    if not recommendation:
        if is_cheaper == True:
            recommendation = f"""Boa notícia! A tarifa {best_code} da {best_provider} permite-lhe poupar {abs(monthly_diff):.2f}€ por mês face à sua fatura atual.

{savings_text}

O processo de mudança é simples e gratuito, demora 2-3 semanas e não há interrupção no fornecimento."""
        elif is_cheaper == False:
            recommendation = f"""A sua tarifa atual com a {current_provider} ({current_bill}€/mês) já é mais competitiva que a melhor alternativa encontrada ({best_total:.2f}€/mês).

{savings_text}

Sugerimos manter a tarifa atual. Se o seu consumo variar ao longo do ano, poderá valer a pena reanalisar noutra altura."""
        else:
            recommendation = f"""A melhor tarifa encontrada é a {best_code} da {best_provider}, com um custo estimado de {best_total:.2f}€/mês para o seu consumo.

O preço médio é de {best_avg:.4f}€/kWh. O processo de mudança é gratuito e sem interrupção no fornecimento."""

    return {
        "success": True,
        "extracted_info": {
            "current_provider": current_provider or "N/A",
            "customer_name": customer or "N/A",
            "customer_nif": nif or "N/A",
            "cpe": cpe or "N/A",
            "contracted_power": f"{power} kVA" if power else "N/A",
            "total_amount_eur": current_bill or "N/A",
        },
        "comparison": {
            "annual_savings": annual_savings or "Ver recomendação",
        },
        "recommendation": recommendation,
        "email_draft": email_draft,
        "llm": llm_used,
    }


# ============================================================
# SCORING LOGIC
# ============================================================

def read_load_csv(content: bytes) -> pd.DataFrame:
    for sep in [",", ";", "\t"]:
        try:
            df = pd.read_csv(io.BytesIO(content), sep=sep)
            if df.shape[1] >= 2: break
        except: continue
    ts_col = next((c for c in df.columns if c.lower() in ["timestamp","ts","datetime","date","data","time"]), df.columns[0])
    energy_col = next((c for c in df.columns if c.lower() in ["kwh","energy_kwh","energy","consumption","consumo"]), df.columns[1])
    df["timestamp"] = pd.to_datetime(df[ts_col], errors="coerce")
    df["energy_kwh"] = pd.to_numeric(df[energy_col].astype(str).str.replace(",","."), errors="coerce")
    result = df[["timestamp","energy_kwh"]].dropna().set_index("timestamp").sort_index()
    try: result.index = result.index.tz_localize("Europe/Lisbon", ambiguous="infer", nonexistent="shift_forward")
    except: pass
    return result


def infer_interval(idx): 
    if len(idx)<2: return 1.0
    d = (idx[1:]-idx[:-1]).to_series().dt.total_seconds().values
    return float(np.median(d[d>0]))/3600 if len(d[d>0]) else 1.0

def choose_kva(peak_kw, pf=0.9, safety=1.1):
    req = (peak_kw/max(pf,1e-6))*safety
    return next((k for k in STANDARD_KVA if k>=req-1e-9), STANDARD_KVA[-1])

def classify_periods(idx, kind):
    dow, hour, weekend = idx.weekday, idx.hour, idx.weekday>=5
    if kind=="flat": return pd.Series(["flat"]*len(idx), index=idx)
    if kind=="tou2": return pd.Series(np.where(weekend|((hour>=22)|(hour<8)),"offpeak","other"), index=idx)
    if kind=="tou3":
        offpeak = weekend|((hour>=22)|(hour<7))
        peak = (~weekend)&(((hour>=9)&(hour<12))|((hour>=19)&(hour<21)))
        return pd.Series(np.where(offpeak,"offpeak",np.where(peak,"peak","shoulder")), index=idx)
    return pd.Series(["flat"]*len(idx), index=idx)

def days_covered(idx): return int(pd.Index(idx.normalize()).nunique())


def normalize_to_30_days(load_df: pd.DataFrame) -> tuple:
    """
    Normalize consumption data to exactly 30 days (a standard month).
    Preserves the hourly consumption pattern for accurate TOU pricing.
    
    Returns: (normalized_df, original_days, original_kwh)
    """
    if load_df.empty:
        return load_df, 0, 0
    
    # Get original stats before normalization
    original_days = days_covered(load_df.index)
    original_kwh = float(load_df["energy_kwh"].sum())
    
    # If already ~30 days (28-31), return as is
    if 28 <= original_days <= 31:
        return load_df, original_days, original_kwh
    
    # Detect the interval (typically 0.25h for 15-min data, 1h for hourly)
    interval_hours = infer_interval(load_df.index)
    readings_per_hour = max(1, int(round(1.0 / interval_hours)))
    
    # Calculate average consumption by hour of day AND day of week (to preserve weekend patterns)
    load_df = load_df.copy()
    load_df["hour"] = load_df.index.hour
    load_df["minute"] = load_df.index.minute
    load_df["dayofweek"] = load_df.index.dayofweek  # 0=Monday, 6=Sunday
    load_df["is_weekend"] = load_df["dayofweek"] >= 5
    
    # Group by: weekend/weekday, hour, minute (for sub-hourly data)
    if readings_per_hour > 1:
        # Sub-hourly data (15-min, 30-min)
        hourly_profile = load_df.groupby(["is_weekend", "hour", "minute"])["energy_kwh"].mean()
    else:
        # Hourly data
        hourly_profile = load_df.groupby(["is_weekend", "hour"])["energy_kwh"].mean()
    
    # Generate 30 days of synthetic data starting from a Monday
    # Use a reference date that starts on Monday for consistent week patterns
    start_date = pd.Timestamp("2024-01-01", tz="Europe/Lisbon")  # This is a Monday
    
    synthetic_data = []
    for day_offset in range(30):
        current_date = start_date + pd.Timedelta(days=day_offset)
        is_weekend = current_date.dayofweek >= 5
        
        for hour in range(24):
            if readings_per_hour > 1:
                # Sub-hourly intervals
                for minute in range(0, 60, int(60 / readings_per_hour)):
                    try:
                        energy = hourly_profile.get((is_weekend, hour, minute), 
                                  hourly_profile.get((is_weekend, hour, 0), 0))
                    except:
                        energy = hourly_profile.mean() if len(hourly_profile) > 0 else 0
                    
                    ts = current_date.replace(hour=hour, minute=minute, second=0)
                    synthetic_data.append({"timestamp": ts, "energy_kwh": float(energy)})
            else:
                # Hourly intervals
                try:
                    energy = hourly_profile.get((is_weekend, hour), hourly_profile.mean())
                except:
                    energy = hourly_profile.mean() if len(hourly_profile) > 0 else 0
                
                ts = current_date.replace(hour=hour, minute=0, second=0)
                synthetic_data.append({"timestamp": ts, "energy_kwh": float(energy)})
    
    # Create normalized dataframe
    normalized_df = pd.DataFrame(synthetic_data)
    normalized_df = normalized_df.set_index("timestamp").sort_index()
    
    return normalized_df, original_days, original_kwh


def score_tariff(load_df, tariff):
    kind, energy, fixed = tariff.get("kind","flat"), tariff.get("energy",{}), float(tariff.get("fixed_eur_day",0))
    d = days_covered(load_df.index)
    if kind=="flat": return fixed*d + float(load_df["energy_kwh"].sum())*float(energy.get("flat",0))
    per, kwh = classify_periods(load_df.index, kind), load_df["energy_kwh"]
    if kind=="tou2": return fixed*d + float(kwh[per=="offpeak"].sum())*float(energy.get("offpeak",0)) + float(kwh[per=="other"].sum())*float(energy.get("other",0))
    if kind=="tou3": return fixed*d + float(kwh[per=="offpeak"].sum())*float(energy.get("offpeak",0)) + float(kwh[per=="shoulder"].sum())*float(energy.get("shoulder",0)) + float(kwh[per=="peak"].sum())*float(energy.get("peak",0))
    return fixed*d + float(load_df["energy_kwh"].sum())*float(next(iter(energy.values()),0))

def load_tariffs():
    if not TARIFFS_PATH.exists(): return []
    data = json.loads(TARIFFS_PATH.read_text(encoding="utf-8"))
    return data.get("tariffs", data) if isinstance(data, dict) else data

def load_links():
    if not OFFERS_CSV_PATH.exists(): return {}
    try:
        df = pd.read_csv(OFFERS_CSV_PATH, sep=";", dtype=str)
        cols = {c.lower():c for c in df.columns}
        return dict(zip(df[cols.get("cod_proposta","")].fillna(""), df[cols.get("linkofertacom","")].fillna(""))) if "cod_proposta" in cols and "linkofertacom" in cols else {}
    except: return {}

def score_tariffs(load_df, power_kva):
    tariffs, links = load_tariffs(), load_links()
    filtered = [t for t in tariffs if abs(float(t.get("power_kva",0))-power_kva)<0.01]
    total_kwh = float(load_df["energy_kwh"].sum())
    results = []
    for t in filtered:
        try:
            total = score_tariff(load_df, t)
            results.append({"id":t.get("id",""), "provider":t.get("provider",""), "cod_proposta":t.get("cod_proposta",""),
                "kind":t.get("kind",""), "kind_label":{"flat":"Simples","tou2":"Bi-horária","tou3":"Tri-horária"}.get(t.get("kind"),t.get("kind")),
                "power_kva":power_kva, "fixed_eur_day":t.get("fixed_eur_day",0), "energy":t.get("energy",{}),
                "total_eur":round(total,2), "avg_price":round(total/total_kwh,4) if total_kwh else 0,
                "offer_link":links.get(t.get("cod_proposta",""),"")})
        except: pass
    results.sort(key=lambda x:x["total_eur"])
    for i,r in enumerate(results): r["rank"]=i+1
    return results

# ============================================================
# FASTAPI APP
# ============================================================

app = FastAPI(title="Otimizador de Tarifas")
session = {"load_df":None, "original_load_df":None, "stats":None, "results":None, "power_kva":None, "analysis":None}

CSS = """
*{box-sizing:border-box}body{font-family:system-ui,sans-serif;margin:0;padding:0;background:linear-gradient(135deg,#667eea,#764ba2);min-height:100vh}
.container{max-width:1200px;margin:0 auto;padding:20px}header{background:#fff;border-radius:12px;padding:20px 30px;margin-bottom:20px;box-shadow:0 4px 6px rgba(0,0,0,.1)}
header h1{margin:0;color:#333}header p{margin:5px 0 0;color:#666}.nav{margin-top:10px}.nav a{color:#667eea;margin-right:20px;text-decoration:none}
.card{background:#fff;border-radius:12px;padding:25px;margin-bottom:20px;box-shadow:0 4px 6px rgba(0,0,0,.1)}.card h2{margin-top:0;color:#333;border-bottom:2px solid #667eea;padding-bottom:10px}
.upload{border:2px dashed #ccc;border-radius:8px;padding:40px;text-align:center;cursor:pointer}.upload:hover{border-color:#667eea;background:#f8f9ff}
.btn{background:linear-gradient(135deg,#667eea,#764ba2);color:#fff;border:none;padding:12px 30px;border-radius:6px;font-size:16px;cursor:pointer;text-decoration:none;display:inline-block}
.btn:hover{transform:translateY(-2px);box-shadow:0 4px 12px rgba(102,126,234,.4)}.btn:disabled{background:#ccc;cursor:not-allowed}.btn-green{background:linear-gradient(135deg,#11998e,#38ef7d)}
.stats{display:grid;grid-template-columns:repeat(auto-fit,minmax(150px,1fr));gap:15px;margin:20px 0}.stat{background:#f8f9ff;border-radius:8px;padding:15px;text-align:center}
.stat .v{font-size:1.8em;font-weight:bold;color:#667eea}.stat .l{color:#666;font-size:.9em}
table{width:100%;border-collapse:collapse;margin-top:15px}th,td{padding:12px;text-align:left;border-bottom:1px solid #eee}th{background:#667eea;color:#fff}tr:hover{background:#f8f9ff}
.best{background:linear-gradient(135deg,#667eea,#764ba2);color:#fff;border-radius:12px;padding:25px;margin-bottom:20px}.best h2{margin-top:0;border-bottom:2px solid rgba(255,255,255,.3);padding-bottom:10px;color:#fff}
.best .grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(150px,1fr));gap:15px;margin-top:15px}.best .item{background:rgba(255,255,255,.15);padding:12px;border-radius:8px}
.best .item .l{font-size:.85em;opacity:.8}.best .item .v{font-size:1.2em;font-weight:bold}
.ai{background:linear-gradient(135deg,#11998e,#38ef7d);color:#fff;border-radius:12px;padding:25px;margin-bottom:20px}.ai h2,.ai h3{color:#fff;margin-top:0}.ai h3{margin-top:20px}
.ai .box{background:rgba(255,255,255,.15);border-radius:8px;padding:20px;margin:15px 0;line-height:1.6}
.email{background:#fff;color:#333;border-radius:8px;padding:20px;margin:15px 0;font-family:monospace;white-space:pre-wrap;max-height:400px;overflow-y:auto}
.copy{background:rgba(255,255,255,.2);color:#fff;border:1px solid rgba(255,255,255,.3);padding:8px 16px;border-radius:4px;cursor:pointer}
.info{display:grid;grid-template-columns:repeat(auto-fit,minmax(200px,1fr));gap:10px;margin:15px 0}.info .i{background:rgba(255,255,255,.1);padding:10px 15px;border-radius:6px}.info .i .l{font-size:.8em;opacity:.8}.info .i .v{font-weight:bold}
.savings{font-size:2em;font-weight:bold;text-align:center;padding:20px;background:rgba(255,255,255,.2);border-radius:8px;margin:15px 0}
.alert{padding:15px;border-radius:8px;margin:15px 0}.alert-warn{background:#fff3cd;color:#856404;border:1px solid #ffc107}.alert-ok{background:#d4edda;color:#155724;border:1px solid #28a745}
.spinner{display:inline-block;width:30px;height:30px;border:3px solid #f3f3f3;border-top:3px solid #667eea;border-radius:50%;animation:spin 1s linear infinite}@keyframes spin{to{transform:rotate(360deg)}}
.kind{display:inline-block;padding:4px 10px;border-radius:12px;font-size:.8em}.kind-flat{background:#e8f5e9;color:#2e7d32}.kind-tou2{background:#e3f2fd;color:#1565c0}.kind-tou3{background:#fce4ec;color:#c2185b}
.rank{display:inline-block;width:28px;height:28px;line-height:28px;text-align:center;border-radius:50%;font-weight:bold}.r1{background:#f5a623;color:#fff}.r2{background:#c0c0c0;color:#fff}.r3{background:#cd7f32;color:#fff}
select{padding:10px 15px;font-size:16px;border:2px solid #ddd;border-radius:6px}
"""

def html(content, title="Otimizador de Tarifas", scripts=""):
    return f"""<!DOCTYPE html><html lang="pt"><head><meta charset="UTF-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>{title}</title><style>{CSS}</style></head>
<body><div class="container"><header><h1>⚡ Otimizador de Tarifas de Eletricidade</h1><p>Compare tarifas e encontre a melhor opção</p>
<div class="nav"><a href="/">🏠 Início</a><a href="/results">📊 Resultados</a><a href="/ai">🤖 Assistente IA</a></div></header>{content}</div>{scripts}</body></html>"""

@app.get("/", response_class=HTMLResponse)
async def home():
    status = get_llm_status()
    has_data = session["load_df"] is not None
    
    stats_html = ""
    if has_data:
        s = session["stats"]
        stats_html = f'''<div class="card"><h2>📈 Dados Carregados</h2>
        <div class="stats"><div class="stat"><div class="v">{s["total_kwh"]:.0f}</div><div class="l">kWh</div></div>
        <div class="stat"><div class="v">{s["days"]}</div><div class="l">Dias</div></div>
        <div class="stat"><div class="v">{s["peak_kw"]:.2f}</div><div class="l">Pico kW</div></div>
        <div class="stat"><div class="v">{s["recommended_kva"]:.2f}</div><div class="l">kVA</div></div></div>
        <a href="/results" class="btn">Ver Resultados →</a> <a href="/ai" class="btn btn-green" style="margin-left:10px">🤖 IA</a></div>'''
    
    ollama_status = f'<span class="alert alert-ok">✅ Ollama activo ({len(status["ollama_models"])} modelos)</span>' if status["ollama_available"] else '<span class="alert alert-warn">⚠️ Ollama não detectado</span>'
    vision_status = f'✅ {status["ollama_vision"]}' if status["ollama_vision"] else "❌ Nenhum (instale: ollama pull llava)"
    
    default_link = '<p style="margin-top:15px;color:#666">💡 Ou <a href="/use-default" style="color:#667eea">usar load.csv existente</a></p>' if DEFAULT_LOAD_PATH.exists() else ""
    
    content = f'''{stats_html}
    <div class="card"><h2>📁 Carregar Consumo</h2><p>CSV com timestamp e kWh</p>
    <form action="/upload" method="post" enctype="multipart/form-data">
    <div class="upload" onclick="document.getElementById('f').click()"><p style="font-size:3em;margin:0">📄</p><p>Clique para selecionar</p>
    <input type="file" id="f" name="file" accept=".csv" required onchange="document.getElementById('btn').disabled=false;document.getElementById('fn').textContent=this.files[0].name" style="display:none"></div>
    <p id="fn" style="margin:10px 0;color:#667eea;font-weight:500"></p>
    <button type="submit" id="btn" class="btn" disabled>Analisar</button></form>{default_link}</div>
    
    <div class="card"><h2>🤖 Assistente IA (GRÁTIS com Ollama)</h2>
    <p>Analise a sua fatura e obtenha email de mudança automático!</p>
    <p><strong>Estado Ollama:</strong> {ollama_status}</p>
    <p><strong>Modelo Visão:</strong> {vision_status}</p>
    <p><strong>Anthropic (alternativa paga):</strong> {"✅ Configurado" if ANTHROPIC_API_KEY else "❌ Não configurado"}</p>
    {'' if status["ollama_available"] else '<div class="alert alert-warn"><strong>Para usar IA grátis:</strong><br>1. Instale Ollama: <a href="https://ollama.ai" target="_blank">ollama.ai</a><br>2. Execute: <code>ollama pull llava</code><br>3. Reinicie esta app</div>'}
    </div>'''
    
    return HTMLResponse(html(content))

@app.get("/use-default")
async def use_default():
    if not DEFAULT_LOAD_PATH.exists():
        return HTMLResponse(html('<div class="card"><h2>❌ Erro</h2><p>load.csv não encontrado</p><a href="/" class="btn">← Voltar</a></div>'))
    return await process_load(DEFAULT_LOAD_PATH.read_bytes())

@app.post("/upload")
async def upload(file: UploadFile = File(...)):
    return await process_load(await file.read())

async def process_load(content: bytes):
    try:
        df = read_load_csv(content)
        if df.empty: raise ValueError("Ficheiro vazio")
        
        # Get original stats before normalization
        original_days = days_covered(df.index)
        original_kwh = float(df["energy_kwh"].sum())
        
        # Calculate peak from original data
        interval = infer_interval(df.index)
        df["kw"] = df["energy_kwh"]/interval
        peak = float(df["kw"].max())
        kva = choose_kva(peak)
        
        # Normalize to 30 days for fair monthly comparison
        normalized_df, _, _ = normalize_to_30_days(df)
        normalized_kwh = float(normalized_df["energy_kwh"].sum())
        
        # Store both original and normalized data
        session["load_df"] = normalized_df  # Use normalized for scoring
        session["original_load_df"] = df    # Keep original for reference
        session["stats"] = {
            "total_kwh": normalized_kwh,           # 30-day normalized consumption
            "days": 30,                             # Always 30 days for comparison
            "peak_kw": peak,
            "recommended_kva": kva,
            "original_days": original_days,         # Original file had X days
            "original_kwh": original_kwh,           # Original total consumption
            "daily_avg_kwh": original_kwh / max(original_days, 1),  # Average daily
        }
        session["power_kva"] = kva
        session["results"] = score_tariffs(normalized_df, kva)
        from fastapi.responses import RedirectResponse
        return RedirectResponse(url="/results", status_code=303)
    except Exception as e:
        return HTMLResponse(html(f'<div class="card"><h2>❌ Erro</h2><p style="color:#c62828">{e}</p><a href="/" class="btn">← Voltar</a></div>'))

@app.get("/results", response_class=HTMLResponse)
async def results(power: Optional[float] = None):
    if session["load_df"] is None:
        from fastapi.responses import RedirectResponse
        return RedirectResponse(url="/")
    
    s, kva = session["stats"], power or session["power_kva"]
    if power and power != session["power_kva"]:
        session["results"] = score_tariffs(session["load_df"], kva)
        session["power_kva"] = kva
    
    res = session["results"]
    if not res:
        return HTMLResponse(html('<div class="card"><h2>⚠️ Sem resultados</h2><a href="/" class="btn">← Voltar</a></div>'))
    
    best = res[0]
    e = best.get("energy", {})
    ep = f"Simples: {e.get('flat',0):.4f}" if best["kind"]=="flat" else f"Vazio: {e.get('offpeak',0):.4f} | Fora: {e.get('other',0):.4f}" if best["kind"]=="tou2" else f"V:{e.get('offpeak',0):.4f} C:{e.get('shoulder',0):.4f} P:{e.get('peak',0):.4f}"
    
    best_html = f'''<div class="best"><h2>🏆 Melhor Tarifa</h2>
    <div class="grid"><div class="item"><div class="l">Comercializador</div><div class="v">{best["provider"]}</div></div>
    <div class="item"><div class="l">Tipo</div><div class="v">{best["kind_label"]}</div></div>
    <div class="item"><div class="l">Total Mensal</div><div class="v">{best["total_eur"]:.2f} €</div></div>
    <div class="item"><div class="l">Médio</div><div class="v">{best["avg_price"]:.4f} €/kWh</div></div></div>
    <p style="margin-top:15px;opacity:.9">{ep} €/kWh</p>
    <div style="margin-top:15px">{"<a href='"+best["offer_link"]+"' target='_blank' style='color:#fff'>Ver oferta →</a>" if best.get("offer_link") else ""} <a href="/ai" style="color:#fff;margin-left:20px">🤖 Gerar email</a></div></div>'''
    
    # Show original data info and normalization explanation
    original_days = s.get("original_days", s["days"])
    original_kwh = s.get("original_kwh", s["total_kwh"])
    daily_avg = s.get("daily_avg_kwh", s["total_kwh"] / 30)
    
    normalization_info = ""
    if original_days != 30:
        normalization_info = f'''<div style="background:#e8f5e9;padding:12px;border-radius:6px;margin-top:15px;border-left:4px solid #4caf50">
        <strong>📅 Dados originais:</strong> {original_days} dias, {original_kwh:.0f} kWh<br>
        <strong>📊 Normalizado para 30 dias</strong> (mês padrão) para comparação justa com a sua fatura mensal.<br>
        <small style="color:#666">Consumo médio diário: {daily_avg:.2f} kWh/dia</small></div>'''
    
    opts = "".join([f'<option value="{k}" {"selected" if abs(k-kva)<0.01 else ""}>{k} kVA</option>' for k in STANDARD_KVA])
    
    rows = ""
    for r in res[:30]:
        rc = f"r{r['rank']}" if r['rank']<=3 else ""
        e = r.get("energy",{})
        et = f"{e.get('flat',0):.4f}" if r["kind"]=="flat" else f"V:{e.get('offpeak',0):.4f} F:{e.get('other',0):.4f}" if r["kind"]=="tou2" else f"V:{e.get('offpeak',0):.3f} C:{e.get('shoulder',0):.3f} P:{e.get('peak',0):.3f}"
        link = f'<a href="{r["offer_link"]}" target="_blank">Ver</a>' if r.get("offer_link") else "-"
        rows += f'<tr><td><span class="rank {rc}">{r["rank"]}</span></td><td><strong>{r["provider"]}</strong><br><small>{r["cod_proposta"]}</small></td><td><span class="kind kind-{r["kind"]}">{r["kind_label"]}</span></td><td><strong>{r["total_eur"]:.2f}€</strong></td><td>{r["avg_price"]:.4f}</td><td style="font-size:.85em">{et}</td><td>{link}</td></tr>'
    
    content = f'''{best_html}
    <div class="card"><h2>📊 Estimativa Mensal: {s["total_kwh"]:.0f} kWh</h2>
    <p>Potência: <select onchange="location.href='/results?power='+this.value">{opts}</select> <span style="color:#666">(Recomendado: {s["recommended_kva"]:.2f} kVA)</span></p>
    {normalization_info}</div>
    <div class="card"><h2>📋 Ranking ({len(res)} tarifas)</h2>
    <p style="color:#666;margin-bottom:15px">Preços calculados para 30 dias - compare diretamente com a sua fatura mensal</p>
    <div style="overflow-x:auto"><table><tr><th>#</th><th>Comercializador</th><th>Tipo</th><th>Total/Mês</th><th>€/kWh</th><th>Energia</th><th>Link</th></tr>{rows}</table></div></div>'''
    
    return HTMLResponse(html(content, "Resultados"))

@app.get("/ai", response_class=HTMLResponse)
async def ai_page():
    if not session["results"]:
        return HTMLResponse(html('<div class="card"><h2>⚠️ Carregue primeiro o consumo</h2><a href="/" class="btn">← Início</a></div>'))
    
    status = get_llm_status()
    best, s = session["results"][0], session["stats"]
    
    prev = ""
    if session.get("analysis") and session["analysis"].get("success"):
        a = session["analysis"]
        ext = a.get("extracted_info", {})
        cmp = a.get("comparison", {})
        # Convert newlines to HTML breaks for display
        rec = a.get("recommendation", "N/A").replace('\n', '<br>')
        email = a.get("email_draft", "N/A")
        prev = f'''<div class="ai"><h2>📄 Análise da Fatura</h2><p style="opacity:.8">via {a.get("llm","LLM")}</p>
        <h3>Dados Extraídos</h3><div class="info">
        <div class="i"><div class="l">Fornecedor Atual</div><div class="v">{ext.get("current_provider","N/A")}</div></div>
        <div class="i"><div class="l">Cliente</div><div class="v">{ext.get("customer_name","N/A")}</div></div>
        <div class="i"><div class="l">NIF</div><div class="v">{ext.get("customer_nif","N/A")}</div></div>
        <div class="i"><div class="l">CPE</div><div class="v">{ext.get("cpe","N/A")}</div></div>
        <div class="i"><div class="l">Potência</div><div class="v">{ext.get("contracted_power","N/A")}</div></div>
        <div class="i"><div class="l">Total Fatura</div><div class="v">{ext.get("total_amount_eur","N/A")}</div></div></div>
        <h3>Poupança</h3><div class="savings">Anual: {cmp.get("annual_savings","N/A")}</div>
        <h3>💡 Recomendação</h3><div class="box">{rec}</div>
        <h3>📧 Email para {best["provider"]}</h3>
        <button class="copy" onclick="navigator.clipboard.writeText(document.getElementById('email').innerText);alert('Copiado!')">📋 Copiar</button>
        <div class="email" id="email">{email}</div></div>'''
    
    vision_warn = "" if status["vision_available"] else '<div class="alert alert-warn">⚠️ Instale modelo de visão: <code>ollama pull llava</code></div>'
    # Check if pymupdf is available for PDF support
    try:
        import fitz
        pdf_support = True
    except ImportError:
        pdf_support = False
    pdf_msg = '<p style="color:#4caf50;font-size:0.9em">✅ PDFs suportados (extração de texto)</p>' if pdf_support else '<p style="color:#ff9800;font-size:0.9em">⚠️ Para PDFs instale: pip install pymupdf</p>'
    
    content = f'''<div class="best"><h2>🏆 Melhor: {best["provider"]} - {best["total_eur"]:.2f}€</h2><p>{best["kind_label"]} | {best["power_kva"]} kVA | {s["days"]} dias</p></div>
    {prev}
    <div class="card"><h2>📝 Inserir Dados Manualmente</h2>
    <p>Preencha os seus dados para gerar o email de mudança. <strong>Não requer modelo de visão!</strong></p>
    <form action="/generate-manual" method="post" id="manualForm">
    <div style="display:grid;grid-template-columns:repeat(auto-fit,minmax(200px,1fr));gap:15px;margin:20px 0">
    <div><label style="display:block;margin-bottom:5px;font-weight:500">Nome Completo *</label><input type="text" name="customer_name" required placeholder="Ex: João Silva" style="width:100%;padding:10px;border:2px solid #ddd;border-radius:6px"></div>
    <div><label style="display:block;margin-bottom:5px;font-weight:500">NIF *</label><input type="text" name="nif" required placeholder="Ex: 123456789" pattern="[0-9]{{9}}" style="width:100%;padding:10px;border:2px solid #ddd;border-radius:6px"></div>
    <div><label style="display:block;margin-bottom:5px;font-weight:500">CPE *</label><input type="text" name="cpe" required placeholder="Ex: PT0002000012345678AA" style="width:100%;padding:10px;border:2px solid #ddd;border-radius:6px"></div>
    <div><label style="display:block;margin-bottom:5px;font-weight:500">Fornecedor Atual</label><select name="current_provider" style="width:100%;padding:10px;border:2px solid #ddd;border-radius:6px">
    <option value="">Selecione...</option><option>EDP</option><option>Iberdrola</option><option>Endesa</option><option>Galp</option><option>Goldenergy</option><option>MEO Energia</option><option>Luzboa</option><option>Repsol</option><option>Plenitude</option><option>SU Eletricidade</option><option>Outro</option></select></div>
    <div><label style="display:block;margin-bottom:5px;font-weight:500">Potência (kVA)</label><select name="power" style="width:100%;padding:10px;border:2px solid #ddd;border-radius:6px">
    <option value="{best['power_kva']}">{best['power_kva']} kVA (selecionado)</option><option value="3.45">3.45 kVA</option><option value="4.60">4.60 kVA</option><option value="5.75">5.75 kVA</option><option value="6.90">6.90 kVA</option><option value="10.35">10.35 kVA</option><option value="13.80">13.80 kVA</option></select></div>
    <div><label style="display:block;margin-bottom:5px;font-weight:500">Valor Última Fatura (€)</label><input type="text" name="current_bill" placeholder="Ex: 75.50" style="width:100%;padding:10px;border:2px solid #ddd;border-radius:6px"></div>
    <div><label style="display:block;margin-bottom:5px;font-weight:500">Email</label><input type="email" name="email" placeholder="seu@email.pt" style="width:100%;padding:10px;border:2px solid #ddd;border-radius:6px"></div>
    <div><label style="display:block;margin-bottom:5px;font-weight:500">Telefone</label><input type="tel" name="phone" placeholder="912345678" style="width:100%;padding:10px;border:2px solid #ddd;border-radius:6px"></div>
    <div style="grid-column:1/-1"><label style="display:block;margin-bottom:5px;font-weight:500">Morada de Fornecimento</label><input type="text" name="address" placeholder="Rua..., Código Postal, Cidade" style="width:100%;padding:10px;border:2px solid #ddd;border-radius:6px"></div>
    </div>
    <button type="submit" class="btn btn-green">📧 Gerar Email de Mudança</button>
    </form></div>
    
    <div class="card"><h2>📄 Analisar Fatura com IA</h2>{vision_warn}
    <p>Carregue a sua fatura (PDF, JPG ou PNG) para extrair dados automaticamente.</p>
    {pdf_msg}
    <form action="/analyze" method="post" enctype="multipart/form-data" id="form">
    <div class="upload" onclick="document.getElementById('r').click()"><p style="font-size:3em;margin:0">🧾</p><p>Selecionar fatura</p>
    <input type="file" id="r" name="receipt" accept=".jpg,.jpeg,.png,.pdf,image/*,application/pdf" required onchange="document.getElementById('btn').disabled=false;document.getElementById('fn').textContent=this.files[0].name" style="display:none"></div>
    <p id="fn" style="margin:10px 0;color:#11998e;font-weight:500"></p>
    <button type="submit" id="btn" class="btn btn-green" disabled>🤖 Analisar com IA</button></form>
    <div id="loading" style="display:none;text-align:center;padding:40px"><div class="spinner"></div><p>A analisar...</p></div></div>
    <div class="card"><h2>💬 Recomendação Rápida</h2><p>Sem dados? Obtenha recomendação baseada apenas no consumo.</p><a href="/recommend" class="btn">Gerar</a></div>'''
    
    scripts = "<script>document.getElementById('form').onsubmit=function(){document.getElementById('loading').style.display='block';document.getElementById('btn').disabled=true}</script>"
    return HTMLResponse(html(content, "Assistente IA", scripts))

@app.post("/analyze")
async def analyze(receipt: UploadFile = File(...)):
    if not session["results"]:
        from fastapi.responses import RedirectResponse
        return RedirectResponse(url="/")
    
    content = await receipt.read()
    
    # Detect actual image format from magic bytes
    if content[:8] == b'\x89PNG\r\n\x1a\n':
        mt = "image/png"
        ext = ".png"
    elif content[:2] == b'\xff\xd8':
        mt = "image/jpeg"
        ext = ".jpg"
    elif content[:4] == b'%PDF':
        mt = "application/pdf"
        ext = ".pdf"
    elif content[:4] == b'GIF8':
        mt = "image/gif"
        ext = ".gif"
    elif content[:4] == b'RIFF' and content[8:12] == b'WEBP':
        mt = "image/webp"
        ext = ".webp"
    else:
        # Try to guess from filename as fallback
        fn = receipt.filename.lower() if receipt.filename else ""
        if fn.endswith('.png'):
            mt, ext = "image/png", ".png"
        elif fn.endswith('.pdf'):
            mt, ext = "application/pdf", ".pdf"
        else:
            mt, ext = "image/jpeg", ".jpg"
    
    result = analyze_receipt_with_llm(content, mt, session["results"][0], session["stats"])
    
    if result.get("error"):
        return HTMLResponse(html(f'<div class="card"><h2>❌ Erro</h2><p>{result["error"]}</p><p style="color:#666;font-size:0.9em">Formato detectado: {mt}</p><a href="/ai" class="btn">← Voltar</a></div>'))
    
    session["analysis"] = result
    from fastapi.responses import RedirectResponse
    return RedirectResponse(url="/ai", status_code=303)

@app.post("/generate-manual")
async def generate_manual(request: Request):
    if not session["results"]:
        from fastapi.responses import RedirectResponse
        return RedirectResponse(url="/")
    
    # Get form data
    form = await request.form()
    user_data = {
        "customer_name": form.get("customer_name", ""),
        "nif": form.get("nif", ""),
        "cpe": form.get("cpe", ""),
        "current_provider": form.get("current_provider", ""),
        "power": form.get("power", ""),
        "current_bill": form.get("current_bill", ""),
        "email": form.get("email", ""),
        "phone": form.get("phone", ""),
        "address": form.get("address", ""),
    }
    
    result = generate_email_from_manual_data(user_data, session["results"][0], session["stats"])
    
    if result.get("error"):
        return HTMLResponse(html(f'<div class="card"><h2>❌ Erro</h2><p>{result["error"]}</p><a href="/ai" class="btn">← Voltar</a></div>'))
    
    session["analysis"] = result
    from fastapi.responses import RedirectResponse
    return RedirectResponse(url="/ai", status_code=303)

@app.get("/recommend")
async def recommend():
    if not session["results"]:
        from fastapi.responses import RedirectResponse
        return RedirectResponse(url="/ai")
    
    result = generate_recommendation(session["results"][0], session["stats"])
    
    if result.get("error"):
        return HTMLResponse(html(f'<div class="card"><h2>❌ Erro</h2><p>{result["error"]}</p><a href="/ai" class="btn">← Voltar</a></div>'))
    
    content = f'''<div class="ai"><h2>💡 Recomendação</h2><p style="opacity:.8">via {result.get("llm","LLM")}</p>
    <div class="box">{result.get("recommendation","").replace(chr(10),"<br>")}</div></div>
    <div class="card"><a href="/ai" class="btn">← Voltar</a></div>'''
    
    return HTMLResponse(html(content, "Recomendação"))

@app.get("/api/status")
async def api_status():
    return get_llm_status()

if __name__ == "__main__":
    status = get_llm_status()
    print("="*50)
    print("⚡ Otimizador de Tarifas de Eletricidade")
    print("="*50)
    print(f"📁 Tariffs: {'✓' if TARIFFS_PATH.exists() else '✗'}")
    print(f"🤖 Ollama: {'✓ '+str(len(status['ollama_models']))+' modelos' if status['ollama_available'] else '✗ não detectado'}")
    print(f"👁️ Visão: {status['ollama_vision'] or '✗ instale: ollama pull llava'}")
    print(f"💰 Anthropic: {'✓' if ANTHROPIC_API_KEY else '✗ (opcional)'}")
    print("="*50)
    print("🌐 http://localhost:8000")
    print("="*50)
    uvicorn.run(app, host="0.0.0.0", port=8000)
