import asyncio
from ai_tutor_bot.agents.tutor_agent import TutorAgent

async def main():
    # Initialize tutor agent
    tutor = TutorAgent()
    
    # Display welcome message
    print("🚀 Welcome to AI Tutor Bot - Comprehensive STEM Assistant!")
    print("--------------------------------------------------------")
    print("📚 Loading knowledge base across all STEM subjects...")
    
    # Comprehensive STEM knowledge base
    documents = [
        # Mathematics
        {
            "id": "math_algebra",
            "source": "Algebra Master",
            "text": "Algebra Fundamentals: Linear equations (y = mx + b), quadratic equations (ax² + bx + c = 0 solved by quadratic formula), polynomials, functions, systems of equations, inequalities, exponents and logarithms."
        },
        {
            "id": "math_calculus",
            "source": "Calculus Compendium",
            "text": "Calculus Principles: Limits, derivatives (power rule, chain rule), integrals (definite and indefinite), fundamental theorem of calculus, series and sequences, multivariable calculus, differential equations."
        },
        {
            "id": "math_geometry",
            "source": "Geometry Encyclopedia",
            "text": "Geometry Concepts: Euclidean geometry, triangles, circles (circumference = 2πr, area = πr²), quadrilaterals, polygons, 3D shapes, coordinate geometry, transformations."
        },
        
        # Physics
        {
            "id": "physics_mechanics",
            "source": "Classical Mechanics",
            "text": "Newton's Laws: 1) Inertia, 2) F=ma, 3) Action-reaction. Kinematics equations: v = u + at, s = ut + ½at², v² = u² + 2as. Conservation laws: energy, momentum, angular momentum."
        },
        {
            "id": "physics_em",
            "source": "Electromagnetism",
            "text": "Maxwell's Equations: Gauss's law, Gauss's law for magnetism, Faraday's law, Ampère's law with Maxwell's addition. Ohm's Law: V=IR. Circuit analysis: series and parallel circuits."
        },
        {
            "id": "physics_quantum",
            "source": "Quantum Physics",
            "text": "Quantum Principles: Wave-particle duality, Schrödinger equation, uncertainty principle. Quantum states, operators, and measurement. Quantum entanglement and superposition."
        },
        
        # Chemistry
        {
            "id": "chem_basics",
            "source": "Chemistry Fundamentals",
            "text": "Atomic Structure: Protons, neutrons, electrons. Periodic table organization. Chemical bonds: ionic, covalent, metallic. Chemical reactions: synthesis, decomposition, combustion."
        },
        {
            "id": "chem_organic",
            "source": "Organic Chemistry",
            "text": "Organic Molecules: Hydrocarbons (alkanes, alkenes, alkynes), functional groups (alcohols, carboxylic acids, amines). Reaction mechanisms: substitution, addition, elimination. Stereochemistry."
        },
        {
            "id": "chem_physical",
            "source": "Physical Chemistry",
            "text": "Thermodynamics: Laws, enthalpy, entropy, Gibbs free energy. Chemical kinetics: reaction rates, rate laws. Quantum chemistry: molecular orbitals, spectroscopy."
        },
        
        # Biology
        {
            "id": "bio_cell",
            "source": "Cell Biology",
            "text": "Cell Structure: Prokaryotic vs eukaryotic cells. Organelles: nucleus, mitochondria, ER, Golgi. Cell division: mitosis and meiosis. Cellular respiration and photosynthesis equations."
        },
        {
            "id": "bio_genetics",
            "source": "Genetics and Evolution",
            "text": "DNA Structure: Double helix, base pairing (A-T, G-C). Central Dogma: DNA → RNA → protein. Mendelian genetics. Evolution: natural selection, genetic drift, speciation."
        },
        {
            "id": "bio_human",
            "source": "Human Physiology",
            "text": "Body Systems: Circulatory (heart, blood vessels), respiratory (lungs, gas exchange), digestive (enzymes, absorption), nervous (neurons, synapses), endocrine (hormones)."
        },
        
        # Computer Science
        {
            "id": "cs_programming",
            "source": "Programming Fundamentals",
            "text": "Programming Concepts: Variables, data types, control structures, functions, OOP principles. Algorithms: sorting, searching, complexity analysis. Data structures: arrays, linked lists, trees, hash tables."
        },
        {
            "id": "cs_ai",
            "source": "Artificial Intelligence",
            "text": "AI Techniques: Machine learning (supervised, unsupervised, reinforcement), neural networks, NLP, computer vision. Ethics in AI. AI applications in various domains."
        }
    ]
    
    # Ingest documents
    await tutor.ingest_documents(documents)
    print("\n✅ Knowledge base loaded with comprehensive coverage of:")
    print("   - Mathematics: Algebra, Calculus, Geometry")
    print("   - Physics: Mechanics, Electromagnetism, Quantum Physics")
    print("   - Chemistry: Fundamentals, Organic, Physical")
    print("   - Biology: Cell Biology, Genetics, Human Physiology")
    print("   - Computer Science: Programming, AI")
    
    # Get user ID
    user_id = input("\n👤 Enter your name: ").strip()
    if not user_id:
        user_id = "Student"
    print(f"👋 Hello {user_id}! Ready to explore STEM?")
    
    # Interactive session
    print("\n💬 Ask ANY question about STEM subjects!")
    print("Type 'exit' to end the session.")
    print("--------------------------------------------------------")
    
    while True:
        query = input("\n❓ Your STEM question: ").strip()
        if not query:
            continue
        if query.lower() in ['exit', 'quit', 'bye']:
            print("\n👋 Goodbye! Keep exploring the wonders of STEM!")
            break
            
        # Process query with visual feedback
        print("🔍 Searching knowledge base...", end='', flush=True)
        response = await tutor.handle_query(user_id, query)
        print("\r", end='')  # Clear searching message
        
        # Display response
        print("\n💡 Answer:", response['answer'])
        print(f"📊 Confidence: {response['relevance_score']:.0%}")
        
        if response['concepts']:
            print("🧠 Key Concepts:", ", ".join(response['concepts'][:3]))
        
        if response['sources']:
            print("📚 Source(s):", ", ".join(set(response['sources'])))
        
        print("--------------------------------------------------------")

if __name__ == "__main__":
    asyncio.run(main())