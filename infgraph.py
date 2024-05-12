import networkx as nx
import matplotlib.pyplot as plt

# Создание пустого графа
G = nx.DiGraph()

# Добавление узлов
G.add_node('Import Libraries')
G.add_node('Global Variables')
G.add_node('Function Definitions')
G.add_node('Main Function')
G.add_node('Main Call')

# Добавление ребер
G.add_edge('Import Libraries', 'Global Variables')
G.add_edge('Global Variables', 'Function Definitions')
G.add_edge('Function Definitions', 'Main Function')
G.add_edge('Main Function', 'Main Call')

# Рисование графа
nx.draw(G, with_labels=True)
plt.show()
