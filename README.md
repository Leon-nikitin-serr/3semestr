Проект: Рекомендательная система для Goodreads

В проекте реализованы три базовых подхода к рекомендациям, их оценка и простые гибридные стратегии для улучшения качества и борьбы с холодным стартом.

Структура данных (ожидаемые CSV)

ratings.csv — поля: book_id, user_id, rating (и др.)

books.csv — метаданные книг (book_id, original_title, authors, average_rating, ratings_count, image_url и т.д.)

tags.csv — tag_id, tag_name

book_tags.csv — goodreads_book_id, tag_id, count

Краткое описание шагов / этапов реализации
Этап 1 — Загрузка и обзор данных

Загружаются файлы ratings, books, tags, book_tags.

Быстрый info() и head() — проверка форматов, пропусков и размеров.

Этап 2 — Baseline (Popularity)

Считаем book_stats = ratings.groupby('book_id').agg(avg_rating, num_ratings)

Фильтр: книги с num_ratings >= 50.

Рекомендуем топ-N по среднему рейтингу (или по ratings_count/popularity).

Использовалось в качестве неперсонализированного baseline.

Этап 3 — Content-based (TF-IDF по тегам/названию)

Собрали текстовый профиль книги: text_profile = original_title + tags.

Векторизовали TF-IDF (TfidfVectorizer(stop_words='english')).

Функция get_similar_books(book_id, N) возвращает похожие книги по косинусной близости TF-IDF.

Этап 4 — Item-based Collaborative Filtering

Построена матрица user_book_rating_matrix = ratings.pivot_table(index='user_id', columns='book_id', values='rating', fill_value=0).

Рассчитана матрица схожестей между книгами (косинус) — item_similarity_test (DataFrame 10k×10k).

Функция predict_rating_item_based(user_id, book_id, ratings_matrix, similarity_matrix, k) вычисляет предсказание через top-k похожих книг (взвешенная сумма).

Протестировали прогноз: предсказанные рейтинги адекватны (~3.9).

Этап 5 — Latent factors (SVD via surprise)

Подготовлены данные для surprise (Dataset.load_from_df с Reader(rating_scale=(1,5))).

Обучили SVD(n_factors=50, n_epochs=20).

Полученный RMSE: 0.8403 на тесте.

Реализована функция get_recommendations(user_id, N) для выдачи top-N по предсказаниям SVD.

Оценка (Precision@K, Recall@K, nDCG@K)

Метрики реализованы как:

precision_at_k(recommended, relevant, k)

recall_at_k(recommended, relevant, k)

ndcg_at_k(recommended, relevant, k)

Релевантными считаются книги с rating >= 4 в тестовой выборке.

Оценка проводится на отложенной тестовой выборке (train_data / test_data), исключая из кандидатов книги, уже увиденные в train.

Фактические результаты :

ItemCF: Precision@5 = 0.022588, Recall@5 = 0.070651, nDCG@5 = 0.071411

SVD: Precision@5 = 0.000867, Recall@5 = 0.000946, nDCG@5 = 0.001171 (RMSE SVD ≈ 0.8403)

Popularity: Precision@5 = 0.000688, Recall@5 = 0.000144, nDCG@5 = 0.000700

Вывод: ItemCF значительно лучше в текущей конфигурации.

Этап 6 — Гибридизация и выводы

Гибридный подход:

На основе анализа метрик (ItemCF показала наилучшие результаты) предлагается использовать гибридную стратегию, объединяющую разные модели:

Для популярных книг, у которых достаточно оценок, применять Item-Based Collaborative Filtering, так как она показывает высокое качество и хорошо работает для часто оцениваемых объектов.

Для новых книг (где нет или мало оценок) использовать контентный подход (TF-IDF по тегам и названиям), который не зависит от пользовательских оценок и позволяет рекомендовать даже при холодном старте.

В качестве резервного варианта можно учитывать популярность книги (например, средний рейтинг или количество оценок).

Такой гибрид компенсирует слабые стороны каждой отдельной модели:

ItemCF обеспечивает персонализацию, Content-based решает проблему новых книг, а Popularity — базовый уровень рекомендаций.

Выводы по работе:

Лучшая модель: ItemCF показала наилучшее качество по всем метрикам (Precision@5, Recall@5, nDCG@5).

Сильные стороны моделей:

ItemCF — точные и персонализированные рекомендации, но зависимость от истории оценок.

SVD — хорошая латентная модель, но требует большого числа данных для обучения.

Popularity — проста и быстра, но не персонализирована.

Content-based — помогает при холодном старте, но не учитывает вкусы пользователя напрямую.

Слабые стороны:

ItemCF и SVD плохо работают для новых пользователей и книг.

Content-based ограничена качеством текстовых описаний и тегов.

Popularity не учитывает индивидуальные предпочтения.
