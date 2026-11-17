from app import build_vector_store, answer_question

vector_store = build_vector_store("C:/Users/Dell/OneDrive/Desktop/Greenfield_Student_Handbook.pdf")

answer, retrieval_context = answer_question(vector_store, "What is the tuition fee per year?")


print("ANSWER:", answer)
print("CHUNKS RETRIEVED:", len(retrieval_context))
print("FIRST CHUNK:", retrieval_context[0][:200])