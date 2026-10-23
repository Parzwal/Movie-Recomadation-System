"use client";

import { useState, useEffect } from 'react';

// API Configuration
const API_BASE_URL = 'http://127.0.0.1:8000/api';

// Types
type Movie = {
  movieId?: number;
  clean_title?: string;
  genres?: string;
  score?: number;
  mean_rating?: number;
  predicted_rating?: number;
  similarity_score?: number;
  Movie?: string;
  "Avg Rating"?: number;
  imdbId?: string;
};

type ChatMessage = {
  role: 'user' | 'ai';
  content: string;
};

type TabType = 'discover' | 'similarity' | 'ai' | 'recommend';

// --- Components ---

function Navbar({ activeTab, setActiveTab }: { activeTab: TabType, setActiveTab: (t: TabType) => void }) {
  return (
    <nav className="fixed w-full top-0 z-50 glass-panel border-b-0 px-8 py-4 flex justify-between items-center transition-all">
      <div className="text-2xl font-black text-netflix-red tracking-tighter cursor-pointer" onClick={() => setActiveTab('discover')}>
        STREAMVIBE
      </div>
      <div className="flex space-x-8 text-sm font-semibold text-gray-400">
        <button 
          onClick={() => setActiveTab('discover')} 
          className={`transition ${activeTab === 'discover' ? 'text-white border-b-2 border-netflix-red pb-1' : 'hover:text-gray-200'}`}
        >
          Discover
        </button>
        <button 
          onClick={() => setActiveTab('recommend')} 
          className={`transition ${activeTab === 'recommend' ? 'text-white border-b-2 border-netflix-red pb-1' : 'hover:text-gray-200'}`}
        >
          Recommend
        </button>
        <button 
          onClick={() => setActiveTab('similarity')} 
          className={`transition ${activeTab === 'similarity' ? 'text-white border-b-2 border-netflix-red pb-1' : 'hover:text-gray-200'}`}
        >
          Similarity Search
        </button>
        <button 
          onClick={() => setActiveTab('ai')} 
          className={`transition ${activeTab === 'ai' ? 'text-white border-b-2 border-netflix-red pb-1' : 'hover:text-gray-200'}`}
        >
          AI Assistant
        </button>
      </div>
    </nav>
  );
}

function HeroBanner() {
  return (
    <div className="relative h-[65vh] w-full bg-cover bg-center flex flex-col justify-end px-12 pb-40" style={{backgroundImage: "url('https://images.unsplash.com/photo-1536440136628-849c177e76a1?q=80&w=2000&auto=format&fit=crop')"}}>
      <div className="absolute inset-0 bg-gradient-to-t from-netflix-black via-netflix-black/50 to-transparent"></div>
      <div className="absolute inset-0 bg-gradient-to-r from-netflix-black/90 via-transparent to-transparent"></div>
    </div>
  );
}

function MovieCard({ movie }: { movie: Movie }) {
  const title = movie.clean_title || movie.Movie || "Unknown Title";
  const genres = movie.genres || "Drama • Thriller";
  
  const posterUrl = `${API_BASE_URL}/poster?title=${encodeURIComponent(title)}${movie.imdbId ? `&imdb_id=${movie.imdbId}` : ''}`;
  const [imgError, setImgError] = useState(false);
  
  return (
    <div className="min-w-[220px] max-w-[220px] rounded-md overflow-hidden bg-[#1e1e1e] group cursor-pointer transition-transform duration-300 hover:scale-105 hover:z-20 shadow-lg border border-white/5 flex flex-col">
      <div className="h-[330px] w-full relative bg-gray-900">
        {!imgError ? (
          <img 
            src={posterUrl} 
            alt={title} 
            className="w-full h-full object-cover" 
            loading="lazy" 
            onError={() => setImgError(true)}
          />
        ) : (
          <div className="w-full h-full flex flex-col items-center justify-center bg-gradient-to-br from-gray-800 to-gray-900 p-4 text-center border border-white/5">
            <span className="text-gray-500 mb-2 text-2xl">🎬</span>
            <span className="text-gray-400 font-bold text-sm leading-tight">{title}</span>
          </div>
        )}
      </div>
      <div className="p-4">
        <h3 className="font-bold text-sm truncate" title={title}>{title}</h3>
        <p className="text-xs text-gray-400 mt-1 truncate">{genres.replace(/\|/g, ' • ')}</p>
        <div className="text-netflix-red font-bold text-xs mt-2">
          {movie.similarity_score ? `${(movie.similarity_score * 100).toFixed(0)}% Match` : 
           movie.score ? `Pop Score: ${movie.score.toFixed(1)}` : 
           movie.mean_rating ? `${movie.mean_rating.toFixed(1)} ⭐` : 
           movie['Avg Rating'] ? `${movie['Avg Rating'].toFixed(1)} ⭐` : 
           movie.predicted_rating ? `Match: ${(movie.predicted_rating * 20).toFixed(0)}%` : 'Trending'}
        </div>
      </div>
    </div>
  );
}

function Carousel({ title, movies }: { title: string, movies: Movie[] }) {
  if (!movies || movies.length === 0) return null;
  
  return (
    <div className="mb-12 relative px-12">
      <h2 className="text-2xl font-bold mb-4">{title}</h2>
      <div className="flex gap-4 overflow-x-auto hide-scrollbar py-4 px-2 -mx-2">
        {movies.map((m, i) => (
          <MovieCard key={i} movie={m} />
        ))}
      </div>
    </div>
  );
}

// --- TABS ---

function DiscoverTab() {
  const [popular, setPopular] = useState<Movie[]>([]);
  const [topRated, setTopRated] = useState<Movie[]>([]);

  useEffect(() => {
    fetch(`${API_BASE_URL}/popular?n=15`)
      .then(r => r.json())
      .then(d => setPopular(d.recommendations || []))
      .catch(e => console.error(e));
      
    fetch(`${API_BASE_URL}/dashboard_stats`)
      .then(r => r.json())
      .then(d => {
        if(d.top_rated) setTopRated(d.top_rated.slice(0, 15));
      })
      .catch(e => console.error(e));
  }, []);

  return (
    <div className="animate-fade-in pb-24">
      <HeroBanner />
      <div className="relative -mt-32 z-20">
        <Carousel title="Top Rated" movies={popular} />
        <Carousel title="Critically Acclaimed Classics" movies={topRated} />
      </div>
    </div>
  );
}

function SimilarityTab() {
  const [query, setQuery] = useState('');
  const [movies, setMovies] = useState<Movie[]>([]);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState('');

  const handleSearch = async () => {
    if (!query.trim()) return;
    setLoading(true);
    setError('');
    setMovies([]);
    
    try {
      const res = await fetch(`${API_BASE_URL}/similar`, {
        method: 'POST',
        headers: {'Content-Type': 'application/json'},
        body: JSON.stringify({ movie_title: query })
      });
      
      if (!res.ok) throw new Error("Movie not found.");
      
      const data = await res.json();
      setMovies(data.recommendations || []);
    } catch (e: any) {
      setError(e.message || "Failed to fetch similar movies.");
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="pt-32 px-12 min-h-screen animate-fade-in pb-24">
      <div className="max-w-4xl mx-auto mb-16 text-center">
        <h1 className="text-5xl font-black mb-4">Find Similar Titles</h1>
        <p className="text-gray-400 text-lg mb-8">Enter your favorite movie and our Item-CF model will find the closest semantic matches in our database.</p>
        
        <div className="flex relative max-w-2xl mx-auto shadow-2xl">
          <input 
            type="text" 
            value={query}
            onChange={(e) => setQuery(e.target.value)}
            onKeyDown={(e) => e.key === 'Enter' && handleSearch()}
            placeholder="e.g. Inception, Toy Story, The Matrix..."
            className="w-full bg-white/10 border border-white/20 rounded-l-lg py-5 pl-8 pr-4 focus:outline-none focus:border-netflix-red text-white text-lg"
          />
          <button 
            onClick={handleSearch}
            className="px-10 bg-netflix-red rounded-r-lg font-bold text-lg hover:bg-red-600 transition"
          >
            {loading ? 'Searching...' : 'Search'}
          </button>
        </div>
        {error && <p className="text-netflix-red mt-4">{error}</p>}
      </div>

      {movies.length > 0 && (
        <div className="max-w-7xl mx-auto">
          <h2 className="text-2xl font-bold mb-6">Because you liked "{query}":</h2>
          <div className="flex flex-wrap gap-6 justify-center">
            {movies.map((m, i) => (
              <MovieCard key={i} movie={m} />
            ))}
          </div>
        </div>
      )}
    </div>
  );
}

function AIChatTab() {
  const [messages, setMessages] = useState<ChatMessage[]>([{role: 'ai', content: 'Hi there! I am your AI movie expert. What kind of vibe are you looking for today?'}]);
  const [input, setInput] = useState('');
  const [loading, setLoading] = useState(false);

  const sendMessage = async () => {
    if (!input.trim() || loading) return;
    
    const userMsg = input.trim();
    setInput('');
    setMessages(prev => [...prev, {role: 'user', content: userMsg}]);
    setLoading(true);
    
    try {
      const res = await fetch(`${API_BASE_URL}/chat`, {
        method: 'POST',
        headers: {'Content-Type': 'application/json'},
        body: JSON.stringify({ message: userMsg, user_id: 42 })
      });
      const data = await res.json();
      setMessages(prev => [...prev, {role: 'ai', content: data.reply}]);
    } catch (e) {
      setMessages(prev => [...prev, {role: 'ai', content: 'Sorry, I am having trouble connecting to my brain!'}]);
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="pt-32 px-12 min-h-screen animate-fade-in flex flex-col items-center pb-24">
      <div className="text-center mb-8">
        <h2 className="text-5xl font-black mb-4">AI Discovery Assistant</h2>
        <p className="text-gray-400 text-lg">Powered by LangGraph, Google Gemini, and FAISS Vector Search.</p>
      </div>
      
      <div className="glass-panel rounded-2xl overflow-hidden w-full max-w-4xl flex flex-col h-[70vh] border border-white/10 shadow-2xl">
        <div className="bg-white/5 p-4 border-b border-white/10 flex items-center">
          <div className="w-10 h-10 rounded-full bg-netflix-red flex items-center justify-center mr-4">
            <svg className="w-6 h-6 text-white" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M13 10V3L4 14h7v7l9-11h-7z" /></svg>
          </div>
          <div>
            <h3 className="font-bold">Vibe AI</h3>
            <p className="text-xs text-gray-400">Online & Ready</p>
          </div>
        </div>
        
        <div className="flex-grow overflow-y-auto p-6 space-y-6 bg-black/20">
          {messages.map((m, i) => (
            <div key={i} className={`flex ${m.role === 'user' ? 'justify-end' : 'justify-start'}`}>
              <div className={`max-w-[80%] rounded-2xl p-5 ${m.role === 'user' ? 'bg-netflix-red text-white rounded-br-none' : 'bg-white/10 text-gray-200 rounded-bl-none border border-white/5'}`}>
                {m.content.split('\n').map((line, j) => (
                   <p key={j} className="mb-2 last:mb-0 leading-relaxed">{line}</p>
                ))}
              </div>
            </div>
          ))}
          {loading && (
             <div className="flex justify-start">
               <div className="bg-white/10 text-gray-200 rounded-2xl p-5 rounded-bl-none border border-white/5 flex space-x-2">
                 <div className="w-2 h-2 bg-gray-400 rounded-full animate-bounce"></div>
                 <div className="w-2 h-2 bg-gray-400 rounded-full animate-bounce" style={{animationDelay: '0.2s'}}></div>
                 <div className="w-2 h-2 bg-gray-400 rounded-full animate-bounce" style={{animationDelay: '0.4s'}}></div>
               </div>
             </div>
          )}
        </div>
        
        <div className="p-4 bg-black/40 border-t border-white/10">
          <div className="flex relative">
            <input 
              type="text" 
              value={input}
              onChange={(e) => setInput(e.target.value)}
              onKeyDown={(e) => e.key === 'Enter' && sendMessage()}
              placeholder="Describe the perfect vibe..."
              className="w-full bg-white/5 border border-white/10 rounded-full py-4 pl-6 pr-16 focus:outline-none focus:border-netflix-red text-white"
            />
            <button 
              onClick={sendMessage}
              className="absolute right-2 top-2 bottom-2 w-10 bg-netflix-red rounded-full flex items-center justify-center hover:bg-red-600 transition"
            >
              <svg className="w-5 h-5 text-white" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M12 19l9 2-9-18-9 18 9-2zm0 0v-8" /></svg>
            </button>
          </div>
        </div>
      </div>
    </div>
  );
}

// --- Main Page ---

function RecommendTab() {
  const [queue, setQueue] = useState<Movie[]>([]);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    // Fetch a large batch of popular movies and shuffle them for a random seed queue
    fetch(`${API_BASE_URL}/popular?n=100`)
      .then(r => r.json())
      .then(d => {
        const movies = d.recommendations || [];
        const shuffled = [...movies].sort(() => Math.random() - 0.5);
        setQueue(shuffled);
        setLoading(false);
      })
      .catch(e => console.error(e));
  }, []);

  const handleLike = async () => {
    if (queue.length === 0) return;
    const currentMovie = queue[0];
    const newQueue = queue.slice(1);
    setQueue(newQueue);
    
    try {
      const queryTitle = currentMovie.clean_title || currentMovie.Movie;
      if(!queryTitle) return;
      const res = await fetch(`${API_BASE_URL}/similar`, {
        method: 'POST',
        headers: {'Content-Type': 'application/json'},
        body: JSON.stringify({ movie_title: queryTitle })
      });
      if (res.ok) {
        const data = await res.json();
        const similar = data.recommendations || [];
        setQueue(prev => {
           const existingIds = new Set(prev.map(m => m.movieId));
           const newUnique = similar.filter((m: Movie) => !existingIds.has(m.movieId));
           return [...newUnique, ...prev];
        });
      }
    } catch (e) {
      console.error(e);
    }
  };

  const handleDislike = () => {
    if (queue.length === 0) return;
    setQueue(prev => prev.slice(1));
  };

  const [imgError, setImgError] = useState(false);

  useEffect(() => {
    // Reset image error state when the movie queue changes
    setImgError(false);
  }, [queue]);

  if (loading) return <div className="pt-32 text-center text-white">Loading...</div>;
  if (queue.length === 0) return <div className="pt-32 text-center text-white">Out of recommendations!</div>;

  const movie = queue[0];
  const title = movie.clean_title || movie.Movie || "Unknown Title";
  const genres = movie.genres || "Drama • Thriller";
  const posterUrl = `${API_BASE_URL}/poster?title=${encodeURIComponent(title)}${movie.imdbId ? `&imdb_id=${movie.imdbId}` : ''}`;

  return (
    <div className="pt-32 px-12 min-h-screen animate-fade-in flex flex-col items-center pb-24">
      <div className="text-center mb-8">
        <h2 className="text-5xl font-black mb-4">Recommended Movie</h2>
        <p className="text-gray-400 text-lg">Teach the algorithm. Like a movie, and we'll instantly queue up similar matches.</p>
      </div>

      <div className="relative w-[400px] h-[600px] rounded-2xl overflow-hidden shadow-2xl bg-[#1e1e1e] border border-white/10 flex flex-col transition-all">
        <div className="h-[480px] w-full relative bg-gray-900 flex items-center justify-center">
          {!imgError ? (
            <img 
              src={posterUrl} 
              alt={title} 
              className="w-full h-full object-cover" 
              onError={() => setImgError(true)}
            />
          ) : (
            <div className="w-full h-full flex flex-col items-center justify-center bg-gradient-to-br from-gray-800 to-gray-900 p-8 text-center border border-white/5">
              <span className="text-6xl mb-4 text-gray-600">🎬</span>
              <span className="text-gray-400 font-black text-2xl leading-tight">{title}</span>
            </div>
          )}
          <div className="absolute inset-0 bg-gradient-to-t from-black via-transparent to-transparent"></div>
        </div>
        
        <div className="p-6 absolute bottom-0 w-full bg-gradient-to-t from-black via-black/95 to-transparent pt-20">
          <h3 className="font-black text-2xl truncate" title={title}>{title}</h3>
          <p className="text-sm text-gray-300 mt-2">{genres.replace(/\|/g, ' • ')}</p>
          
          <div className="flex justify-between items-center mt-6">
            <button 
              onClick={handleDislike}
              className="w-16 h-16 rounded-full bg-white/10 flex items-center justify-center hover:bg-white/20 hover:text-white transition-colors border border-white/20 text-gray-400"
            >
              <svg className="w-8 h-8" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M6 18L18 6M6 6l12 12" /></svg>
            </button>
            <button 
              onClick={handleLike}
              className="w-16 h-16 rounded-full bg-netflix-red flex items-center justify-center hover:bg-red-600 transition-colors border border-netflix-red/50 shadow-lg shadow-netflix-red/30"
            >
              <svg className="w-8 h-8 text-white" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M4.318 6.318a4.5 4.5 0 000 6.364L12 20.364l7.682-7.682a4.5 4.5 0 00-6.364-6.364L12 7.636l-1.318-1.318a4.5 4.5 0 00-6.364 0z" /></svg>
            </button>
          </div>
        </div>
      </div>
    </div>
  );
}

export default function Home() {
  const [activeTab, setActiveTab] = useState<TabType>('discover');

  return (
    <div className="min-h-screen bg-netflix-black transition-colors duration-500">
      <Navbar activeTab={activeTab} setActiveTab={setActiveTab} />
      
      {activeTab === 'discover' && <DiscoverTab />}
      {activeTab === 'recommend' && <RecommendTab />}
      {activeTab === 'similarity' && <SimilarityTab />}
      {activeTab === 'ai' && <AIChatTab />}
    </div>
  );
}
