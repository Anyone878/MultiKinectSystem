#pragma once

#include <mutex>
#include <condition_variable>
#include <deque>
#include <vector>

namespace My {
	template<typename T>
	class ThreadSafeQueue {
	public:
		explicit ThreadSafeQueue(size_t max_size)
			: max_size_(max_size) {}

		// Copying and assignment prohibited
		ThreadSafeQueue(const ThreadSafeQueue&) = delete;
		ThreadSafeQueue& operator=(const ThreadSafeQueue&) = delete;

		void push(const T& item) {
			std::unique_lock<std::mutex> lock(mutex_);

			if (queue_.size() >= max_size_) {
				queue_.pop_front();
			}

			queue_.push_back(item);
			cond_var_.notify_one();
		}

		void push(T&& item) {
			std::unique_lock<std::mutex> lock(mutex_);

			if (queue_.size() >= max_size_) {
				queue_.pop_front();
			}

			queue_.push_back(std::move(item));
			cond_var_.notify_one();
		}

		T pop() {
			std::unique_lock<std::mutex> lock(mutex_);
			cond_var_.wait(lock, [this] { return !queue_.empty(); });

			T item = std::move(queue_.front());
			queue_.pop_front();
			return item;
		}

		bool try_pop(T& item) {
			std::unique_lock<std::mutex> lock(mutex_);
			if (queue_.empty()) {
				return false;
			}

			item = std::move(queue_.front());
			queue_.pop_front();
			return true;
		}

		size_t size() const {
			std::lock_guard<std::mutex> lock(mutex_);
			return queue_.size();
		}

		bool empty() const {
			std::lock_guard<std::mutex> lock(mutex_);
			return queue_.empty();
		}

		size_t capacity() const {
			return max_size_;
		}

	private:
		mutable std::mutex mutex_;
		std::condition_variable cond_var_;
		std::deque<T> queue_;
		size_t max_size_;
	};
}