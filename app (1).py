import streamlit as st
import torch
import torch.nn as nn
import sentencepiece as spm

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# --- Original Modular Architecture ---
class Encoder(nn.Module):
    def __init__(self, vocab_size, emb_dim=256, enc_hid_dim=512, dec_hid_dim=512):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, emb_dim)
        self.rnn = nn.LSTM(emb_dim, enc_hid_dim, num_layers=2, bidirectional=True, batch_first=True)
        self.fc_hidden = nn.Linear(enc_hid_dim * 2, dec_hid_dim)
        self.fc_cell = nn.Linear(enc_hid_dim * 2, dec_hid_dim)

    def forward(self, src):
        embedded = self.embedding(src)
        enc_outs, (hidden, cell) = self.rnn(embedded)
        # Combine 2 layers of bidirectional states
        h_cat = torch.cat([hidden[0:4:2], hidden[1:4:2]], dim=2)
        c_cat = torch.cat([cell[0:4:2], cell[1:4:2]], dim=2)
        dec_h = torch.tanh(self.fc_hidden(h_cat))
        dec_c = torch.tanh(self.fc_cell(c_cat))
        return enc_outs, (dec_h, dec_c)

class Attention(nn.Module):
    def __init__(self, enc_hid_dim=512, dec_hid_dim=512):
        super().__init__()
        self.attn = nn.Linear((enc_hid_dim * 2) + dec_hid_dim, dec_hid_dim)
        self.v = nn.Linear(dec_hid_dim, 1, bias=False)

    def forward(self, hidden, encoder_outputs):
        # hidden from top layer
        top_h = hidden[-1].unsqueeze(1).repeat(1, encoder_outputs.shape[1], 1)
        energy = torch.tanh(self.attn(torch.cat((top_h, encoder_outputs), dim=2)))
        attention = torch.softmax(self.v(energy).squeeze(2), dim=1)
        return attention

class Decoder(nn.Module):
    def __init__(self, vocab_size, emb_dim=256, enc_hid_dim=512, dec_hid_dim=512):
        super().__init__()
        self.attention = Attention(enc_hid_dim, dec_hid_dim)
        self.embedding = nn.Embedding(vocab_size, emb_dim)
        self.rnn = nn.LSTM((enc_hid_dim * 2) + emb_dim, dec_hid_dim, num_layers=2, batch_first=True)
        self.fc_out = nn.Linear((enc_hid_dim * 2) + dec_hid_dim + emb_dim, vocab_size)

    def forward(self, input, hidden, cell, encoder_outputs):
        input = input.unsqueeze(1)
        embedded = self.embedding(input)
        a = self.attention(hidden, encoder_outputs)
        context = torch.bmm(a.unsqueeze(1), encoder_outputs)
        rnn_input = torch.cat((embedded, context), dim=2)
        output, (hidden, cell) = self.rnn(rnn_input, (hidden, cell))
        prediction = self.fc_out(torch.cat((output.squeeze(1), context.squeeze(1), embedded.squeeze(1)), dim=1))
        return prediction, hidden, cell

class Seq2Seq(nn.Module):
    def __init__(self, encoder, decoder):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder

@st.cache_resource
def load_components():
    sp = spm.SentencePieceProcessor(model_file="ur_sp.model")
    vocab_size = len(sp)
    
    # Instantiate the modular architecture
    enc = Encoder(vocab_size)
    dec = Decoder(vocab_size)
    model = Seq2Seq(enc, dec)
    
    # Load weights
    checkpoint = torch.load("best_model.pt", map_location=device)
    state_dict = checkpoint["model_state_dict"] if "model_state_dict" in checkpoint else checkpoint
    model.load_state_dict(state_dict)
    
    model.to(device)
    model.eval()
    return sp, model

sp, model = load_components()

# --- Decoding Logic with Repetition Control ---
def apply_repetition_suppression(logits, generated_tokens, penalty=1.35, block_ngram=2):
    for token_id in set(generated_tokens):
        if logits[0, token_id] > 0:
            logits[0, token_id] /= penalty
        else:
            logits[0, token_id] *= penalty

    if len(generated_tokens) >= block_ngram:
        prefix = tuple(generated_tokens[-(block_ngram - 1):])
        for i in range(len(generated_tokens) - block_ngram + 1):
            if tuple(generated_tokens[i:i + block_ngram - 1]) == prefix:
                forbidden_token = generated_tokens[i + block_ngram - 1]
                logits[0, forbidden_token] = -float("inf")
    return logits

def greedy_generate(src_text, max_len=30):
    src_ids = [sp.bos_id()] + sp.encode(src_text) + [sp.eos_id()]
    src_tensor = torch.tensor([src_ids], dtype=torch.long, device=device)

    with torch.no_grad():
        enc_outs, (h, c) = model.encoder(src_tensor)

    generated = []
    current_token = sp.bos_id()
    
    for _ in range(max_len):
        cur_tensor = torch.tensor([current_token], dtype=torch.long, device=device)
        logits, h, c = model.decoder(cur_tensor, h, c, enc_outs)

        logits = apply_repetition_suppression(logits, generated)
        next_token = torch.argmax(logits, dim=-1).item()

        if next_token == sp.eos_id():
            break
        generated.append(next_token)
        current_token = next_token

    return sp.decode(generated)

def beam_search_generate(src_text, beam_width=3, max_len=30):
    src_ids = [sp.bos_id()] + sp.encode(src_text) + [sp.eos_id()]
    src_tensor = torch.tensor([src_ids], dtype=torch.long, device=device)

    with torch.no_grad():
        enc_outs, (h, c) = model.encoder(src_tensor)

    hypotheses = [(0.0, [], (h, c))]

    for _ in range(max_len):
        all_candidates = []
        for score, seq, (dec_h, dec_c) in hypotheses:
            if seq and seq[-1] == sp.eos_id():
                all_candidates.append((score, seq, (dec_h, dec_c)))
                continue

            last_id = seq[-1] if seq else sp.bos_id()
            cur_tensor = torch.tensor([last_id], dtype=torch.long, device=device)
            
            logits, new_h, new_c = model.decoder(cur_tensor, dec_h, dec_c, enc_outs)

            logits = apply_repetition_suppression(logits, seq)
            log_probs = torch.log_softmax(logits, dim=-1).squeeze(0)

            top_scores, top_indices = torch.topk(log_probs, beam_width)
            for s, idx in zip(top_scores.tolist(), top_indices.tolist()):
                all_candidates.append((score + s, seq + [idx], (new_h, new_c)))

        hypotheses = sorted(all_candidates, key=lambda x: x[0], reverse=True)[:beam_width]
        if all(seq and seq[-1] == sp.eos_id() for _, seq, _ in hypotheses):
            break

    best_seq = hypotheses[0][1]
    cleaned_seq = [t for t in best_seq if t != sp.eos_id()]
    return sp.decode(cleaned_seq)

# --- Streamlit Front End ---
st.title("PK Urdu AI Teacher: Question Generator")
st.write("Paste an Urdu sentence below and wrap the target answer in `<ans>` and `</ans>`. The AI will generate the appropriate question.")

user_text = st.text_area(
    "Input Urdu Sentence:",
    "شہر حاصل کی۔ <ans> کی دہائی کے آخر میں 1990 </ans> بیوسٹن ، ٹیکساس میں پیدا ہوئی ، اور"
)

if st.button("Generate Question"):
    with st.spinner("Generating..."):
        g_out = greedy_generate(user_text)
        b_out = beam_search_generate(user_text, beam_width=3)

    st.success("Generation Complete!")
    
    st.subheader("Greedy Search Output:")
    st.info(g_out if g_out.strip() else "کوئی نتیجہ نہیں مل سکا۔")

    st.subheader("Beam Search Output (k=3):")
    st.info(b_out if b_out.strip() else "کوئی نتیجہ نہیں مل سکا۔")