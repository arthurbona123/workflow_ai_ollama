# Dados sobre testes realizados sobre inferência e performance em modelos SLMs (<=2B)

* O primeiro teste realizado foi sobre o script `test_step1_chamada_invoke.py`, em que consiste em apenas carregar o modelo e trocar mensagens com ele, sem ferramentas, sem definições forçadas de parâmetros de inferência ou SystemMessages() e apenas utilizando o comando direto não-estruturado `.invoke()`.

## Dados sobre modelos, prompts e configurações

* Os modelos que irão ser testados são:

1. gemma-2-2b-it
2. qwen/qwen3-1.7b
3. mistralai/ministral-3-3b
4. smollm2-1.7b-instruct

* O System Prompt envolvido em cada um será resumido, pequeno e objetivo, em Inglês, com perguntas nos prompts em Inglês também, uma vez que já foi constatado
a grande diferença de performance e de qualidade nas respostas:

```text
# ROLE: SENIOR SOFTWARE ENGINEER
- You are a Sênior Software Engineer with so much experience in the area, especialized in code review and fix, code development, Q&A of questions about software and programming-related points
and general support for the user.
```

* Os parametros de inferência foram calibrados seguindo os estudos e levantamento de dados sobre como cada um afeta a geração e como muitos deles já se encontram obsoletos
para obter uma boa performance e filtragem ideal de tokens:

| Parâmetro | Valor |
| --- | --- |
| temperature | 0.3 |
| top_k | 30 |
| top_p | 0.90 |
| min_p | 0.05 |
| num_ctx | 4096 |

* O prompt enviado para a análise sobre como foi a performance deles será este:

```text
When working with collections in Python, transforming and filtering data efficiently is a frequent task. 
Suppose you have a list of integers, and you need to generate a new list containing only the even numbers from the original list, with each of those numbers squared. 
Explain how to accomplish this using both a traditional for loop with an append() method and a list comprehension. 
Compare the two approaches in terms of code readability, conciseness, and Pythonic style, and briefly describe how to include an if-else conditional expression directly inside a list comprehension. 
```

## Teste de Performance e Resultados análisados por Modelo

1. gemma-2-2b-it: Se demonstrou muito eficiente e respondeu com clareza a pergunta feita. Trouxe exemplos de códigos claros, sem alucinações e invenções de snippets, 
e não demonstrou o uso de tokens desnecessários ou excessivos, como por exemplo repetições incorretas, frases copiadas e etc. Sobre o dominio da linguagem Python, demonstrou bom 
entendimento e foco na atenção da construção do código, em que tudo segue boas práticas e são códigos rodáveis e coerentes com o pedido do prompt inicial.

2. qwen/qwen3-1.7b: Se demonstrou muito eficiente e respondeu com clareza e exatidão a pergunta feita. Não houve alucinações e nem falta de entendimento ou de atenção sobre os elementos solicitados e perguntados no 
prompt, em que respondeu usando snippets corretos e bem escritos de código em Python, explicações em Markdown claras e didáticas e abordagens até mais completas que o `gemma-2-2b-it`. 
Seu tempo de inferencia foi relativamente maior por conta de seu suporte a reasoning, que foi feito com sucesso e precisão, o qual está limitado à 1024 tokens de output no máximo.

3. mistralai/ministral-3-3b: Demonstrou performance muito eficiente e muito atenta para um modelo sem reasoning. Ele conseguir responder a pergunta com clareza, usando elementos Markdown como forma de organizar as coisas, 
e utilizando snippets corretos e bem escritos de código em Python, em que trouxe abordagens de self-revision muito interessantes. Forçando a resposta para números impares serem elevados ao cubo, o modelo retornou 
perfeitamente a resposta e trouxe uma explicação técnica e didática sobre o que foi feito e o por que, pensando na estrutura da linguagem Python.

4. smollm2-1.7b-instruct: Não demonstrou performance agradável para o que se é esperado em relação aos modelos de mesmo tamanho e relativo 'treinamento', alucinando e repetindo tokens ao longo das escritas do código Python, 
fora que quebrou sintaxe na maioria das vezes e não respondeu corretamente a pergunta de forma concisa e coerente, muito menos organizada.