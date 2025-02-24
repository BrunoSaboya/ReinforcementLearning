# Comparação entre Q-Learning e SARSA nos ambientes Taxi-Driver e Cliff Walking

## 1. Resultados no Ambiente Taxi-Driver

**Qual algoritmo tem os melhores resultados para o ambiente do Taxi-Driver?**
No ambiente Taxi-Driver, o algoritmo Q-Learning apresentou resultados ligeiramente melhores em termos de recompensa média após o treinamento. Isso se deve ao fato do Q-Learning ser um algoritmo off-policy, o que permite que ele aprenda a política ótima mesmo quando o agente não segue estritamente essa política durante o treinamento.

**A curva de aprendizado dos dois algoritmos é a mesma?**
As curvas de aprendizado apresentam padrões similares, mas o Q-Learning geralmente converge mais rapidamente para uma política ótima. O SARSA, por ser on-policy, tende a ter uma curva de aprendizado mais suave, especialmente em cenários onde o agente explora mais o ambiente.

**O comportamento final do agente, depois de treinado, é ótimo em ambos os casos?**
Sim, ambos os agentes aprenderam a executar a tarefa de forma eficiente, mas o agente treinado com Q-Learning demonstrou um comportamento mais próximo do ótimo em cenários mais variados, enquanto o agente SARSA foi mais seguro e evitou decisões arriscadas.

## 2. Resultados no Ambiente Cliff Walking

**Qual algoritmo tem os melhores resultados para o ambiente do Cliff Walking?**
No ambiente Cliff Walking, o SARSA apresentou um desempenho mais estável e evitou com mais frequência cair no "penhasco". O Q-Learning, apesar de potencialmente alcançar um caminho mais curto e recompensas mais altas, também tomou mais decisões arriscadas, o que resultou em punições mais severas em alguns episódios.

**A curva de aprendizado dos dois algoritmos é a mesma?**
Não, a curva de aprendizado do SARSA é mais conservadora, mostrando menos variação negativa. O Q-Learning, devido ao seu comportamento mais otimista, teve picos e vales mais acentuados.

**O comportamento final do agente, depois de treinado, é ótimo?**
O comportamento do Q-Learning pode ser considerado ótimo em termos de buscar a recompensa máxima, mas isso vem com o risco de punições altas no Cliff Walking. O SARSA, embora possa demorar mais para atingir o caminho ótimo, foi mais seguro e consistente.

**Qual agente tem um comportamento mais conservador e qual tem um comportamento mais otimista?**
O SARSA tem um comportamento mais conservador, pois leva em consideração a política atual e a ação real tomada pelo agente. O Q-Learning é mais otimista, já que sempre considera a ação de maior valor potencial, independentemente da ação que o agente efetivamente tomará.

## 3. Diferença Geral entre Q-Learning e SARSA

**De uma forma geral, qual seria a diferença entre os algoritmos Q-Learning e SARSA?**
A principal diferença entre os dois algoritmos é que o Q-Learning é off-policy, o que significa que ele aprende com base na melhor ação possível, mesmo que o agente não execute essa ação. O SARSA é on-policy, ou seja, ele aprende a partir das ações reais tomadas pelo agente, seguindo a política atual.

**Os agentes treinados teriam o mesmo comportamento?**
Não necessariamente. O agente treinado com Q-Learning tende a ser mais explorador e pode alcançar uma política ótima mais rapidamente, mas com riscos maiores. O agente treinado com SARSA será mais cauteloso, especialmente em ambientes com penalidades severas (como o Cliff Walking).

**As curvas de aprendizado também seriam iguais?**
Não. O Q-Learning geralmente apresenta uma curva de aprendizado mais agressiva, com picos e vales mais acentuados. O SARSA tem uma curva mais suave e gradual, refletindo seu comportamento conservador durante o treinamento.

